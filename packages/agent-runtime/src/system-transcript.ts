import { isDeepStrictEqual } from "node:util";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import {
  contentText,
  createInitialSystemMessage,
  getCurrentSystemMessage,
  getCurrentSystemPrompt,
  getCurrentTools,
  getToolStateChanges,
  type SystemMessage,
  type Tool,
  toToolDeclaration,
} from "@earendil-works/pi-ai";

type DynamicSystemKind = "resumable" | "transient";
// Runtime ownership stays out of persisted history and provider payloads.
const dynamicSystemKinds = new WeakMap<AgentMessage, DynamicSystemKind>();

function baseSystemMessages(messages: readonly AgentMessage[]): SystemMessage[] {
  return systemMessages(messages).filter((message) => !dynamicSystemKinds.has(message));
}

function currentDynamicMessages(messages: readonly AgentMessage[]): AgentMessage[] {
  const latest = new Map<DynamicSystemKind, AgentMessage>();
  for (const message of messages) {
    const kind = dynamicSystemKinds.get(message);
    if (kind) latest.set(kind, message);
  }
  return messages.filter((message) => {
    const kind = dynamicSystemKinds.get(message);
    return kind !== undefined && latest.get(kind) === message;
  });
}

function systemMessages(messages: readonly AgentMessage[]): SystemMessage[] {
  return messages.filter((message): message is SystemMessage => message.role === "system");
}

function nonSystemMessages(messages: readonly AgentMessage[]): AgentMessage[] {
  return messages.filter((message) => message.role !== "system");
}

function nextSystemTimestamp(messages: readonly AgentMessage[]): number {
  // Same-millisecond updates still have to land after the last response;
  // a clock rewind only moves forward and never predates existing usage.
  return messages.reduce((timestamp, message) => Math.max(timestamp, message.timestamp + 1), Date.now());
}

function currentSystemMessage(messages: readonly AgentMessage[]): SystemMessage | undefined {
  const systems = systemMessages(messages);
  if (systems.length < 2) return systems[0];
  const current = getCurrentSystemMessage(systems)!;
  // Upstream folding keeps the first row's timestamp, but section/tool
  // deletions may come from a later delta. The snapshot's semantic time must
  // cover every contributor or old usage would be wrongly reactivated.
  return {
    ...current,
    timestamp: systems.reduce((timestamp, message) => Math.max(timestamp, message.timestamp), systems[0]!.timestamp),
  };
}

/** The unrendered content, without flattening named sections into the prompt. */
export function systemPromptContent(messages: readonly AgentMessage[]): string {
  return contentText(getCurrentSystemMessage(baseSystemMessages(messages))?.content ?? "");
}

export function initialSystemTranscript(
  prompt: string,
  tools: readonly Tool[],
  messages: AgentMessage[],
): AgentMessage[] {
  const system = createInitialSystemMessage(prompt, tools.map(toToolDeclaration));
  // Persisted history does not record system state, so a restart cannot prove
  // the new config matches an old request. Use the real init time; never keep
  // the upstream initial timestamp 0 that would let historical usage pretend
  // it is still valid.
  return system
    ? [{ ...system, timestamp: nextSystemTimestamp(messages) }, ...messages]
    : messages;
}

export function replaceSystemPrompt(messages: AgentMessage[], prompt: string): AgentMessage[] {
  const base = baseSystemMessages(messages);
  const current = currentSystemMessage(base);
  if (prompt === getCurrentSystemPrompt(base) || prompt === contentText(current?.content ?? "")) {
    return messages;
  }
  // prompt replaces content only; named sections and the final tool state stay
  // with upstream replay. Recovery reads unrendered content so sections are
  // not embedded into content a second time.
  return [
    { ...current, role: "system", content: prompt, timestamp: nextSystemTimestamp(messages) },
    ...messages.filter((message) => message.role !== "system"),
    ...currentDynamicMessages(messages),
  ];
}

/**
 * Append current dynamic system text after the already-emitted transcript.
 * Used for resumable-list updates and one-shot recovery nudges so the head
 * system message and prior history stay byte-identical for prefix cache.
 */
export function appendSystemMessage(
  messages: AgentMessage[],
  content: string,
  kind: DynamicSystemKind = "resumable",
): AgentMessage[] {
  if (!content) return messages;
  const last = currentDynamicMessages(messages).find((message) => dynamicSystemKinds.get(message) === kind);
  if (last?.role === "system" && contentText(last.content ?? "") === content) return messages;
  const row: SystemMessage = { role: "system", content, timestamp: nextSystemTimestamp(messages) };
  dynamicSystemKinds.set(row, kind);
  return [...messages, row];
}

/** Provider adapters fold all system rows into the head. Keep runtime-owned
 * status and recovery context in place on the wire instead of folding it. */
export function projectDynamicSystemMessages(messages: AgentMessage[]): AgentMessage[] {
  const projected: AgentMessage[] = [];
  const pending = new Set<string>();
  const deferred: AgentMessage[] = [];
  for (const message of messages) {
    if (message.role === "system" && dynamicSystemKinds.has(message)) {
      const row: AgentMessage = {
        role: "user",
        content: `<runtime-context>\n${contentText(message.content ?? "")}\n</runtime-context>`,
        timestamp: message.timestamp,
      };
      (pending.size ? deferred : projected).push(row);
      continue;
    }
    if (message.role === "assistant") {
      for (const block of message.content) {
        if (block.type === "toolCall") pending.add(block.id);
      }
    } else if (message.role === "toolResult") {
      pending.delete(message.toolCallId);
    } else if (message.role === "user") {
      // Real user interruptions retain the SDK's normal missing-result policy.
      pending.clear();
    }
    projected.push(message);
    if (pending.size === 0 && deferred.length) projected.push(...deferred.splice(0));
  }
  // Unfinished batches keep their dynamic rows internal until results arrive;
  // a synthetic user row here would make the SDK invent duplicate tool results.
  return projected;
}

function isNonSystemExtension(
  previousNonSystem: readonly AgentMessage[],
  messages: readonly AgentMessage[],
): boolean {
  if (messages.length < previousNonSystem.length) return false;
  for (let index = 0; index < previousNonSystem.length; index += 1) {
    const previous = previousNonSystem[index];
    if (messages[index] !== previous && !isDeepStrictEqual(messages[index], previous)) return false;
  }
  return true;
}

export function rebuildSystemTranscript(
  previous: readonly AgentMessage[],
  messages: AgentMessage[],
): AgentMessage[] {
  // A live transcript already carries system rows in place, including tool
  // deltas and trailing appends. Reusing it avoids folding those rows into a
  // new head message, which would rewrite the emitted request prefix.
  if (messages.some((message) => message.role === "system")) return messages;
  const systems = systemMessages(previous);
  if (systems.length === 0) return messages;
  const previousNonSystem = nonSystemMessages(previous);
  if (isNonSystemExtension(previousNonSystem, messages)) {
    if (messages.length === previousNonSystem.length) return previous as AgentMessage[];
    const rebuilt: AgentMessage[] = [];
    let index = 0;
    for (const message of previous) {
      if (message.role === "system") {
        rebuilt.push(message);
        continue;
      }
      rebuilt.push(message);
      index += 1;
    }
    while (index < messages.length) {
      rebuilt.push(messages[index]!);
      index += 1;
    }
    return rebuilt;
  }
  // Compaction and other rewrites are a new prefix. Fold the effective system
  // snapshot in front; folding does not invent a timestamp newer than the
  // contributing system rows.
  const system = currentSystemMessage(baseSystemMessages(previous));
  return [...(system ? [system] : []), ...messages, ...currentDynamicMessages(previous)];
}

export function syncSystemTools(messages: AgentMessage[], tools: readonly Tool[]): AgentMessage[] {
  const changes = getToolStateChanges(getCurrentTools(messages), tools);
  if (changes.toolsAdded.length === 0 && changes.toolsRemoved.length === 0) return messages;
  // A real tool change becomes a newer prefix, so old usage is invalidated.
  // Keep add/remove/same-name-replace deltas and keep the declared set aligned
  // with the executable catalog so the agent loop does not append the same
  // change again on the next turn.
  return [
    ...systemMessages(messages),
    {
      role: "system",
      content: "",
      ...(changes.toolsAdded.length ? { toolsAdded: changes.toolsAdded } : {}),
      ...(changes.toolsRemoved.length ? { toolsRemoved: changes.toolsRemoved } : {}),
      timestamp: nextSystemTimestamp(messages),
    },
    ...messages.filter((message) => message.role !== "system"),
  ];
}
