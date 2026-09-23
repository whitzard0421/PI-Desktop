import { afterEach, describe, expect, it, vi } from "vitest";
import { Agent, convertToLlm, type AgentMessage, type AgentTool } from "@earendil-works/pi-agent-core";
import {
  createAssistantMessageEventStream,
  getCurrentSystemMessage,
  getCurrentSystemPrompt,
  getCurrentTools,
  toToolDeclaration,
  Type,
  type AssistantMessage,
  type SystemMessage,
  type Tool,
} from "@earendil-works/pi-ai";
import { estimateContextTokens as estimateTranscriptTokens } from "@earendil-works/pi-ai/utils/estimate";
import {
  appendSystemMessage,
  initialSystemTranscript,
  projectDynamicSystemMessages,
  rebuildSystemTranscript,
  replaceSystemPrompt,
  syncSystemTools,
  systemPromptContent,
} from "./system-transcript.js";


const read: Tool = { name: "Read", description: "Read text", parameters: Type.Object({ path: Type.String() }) };
const edit: Tool = { ...read, name: "Edit", description: "Edit text" };
const initial: SystemMessage = {
  role: "system", content: "Base instructions", timestamp: 1_000,
  sections: { rules: "Keep these rules", obsolete: "Remove these rules" },
  toolsAdded: [read, edit],
};
const delta: SystemMessage = {
  role: "system", content: "Additional instructions", timestamp: 1_500,
  sections: { rules: "Updated rules", obsolete: null },
  toolsRemoved: [{ name: "Edit" }],
};
const assistant: AssistantMessage = {
  role: "assistant", content: [{ type: "text", text: "done" }],
  api: "openai-completions", provider: "local", model: "local-model", stopReason: "stop", timestamp: 2_000,
  usage: {
    input: 990, output: 10, cacheRead: 0, cacheWrite: 0, totalTokens: 1_000,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  },
};
const user: AgentMessage = { role: "user", content: "continue", timestamp: 2_100 };

function estimateContextTokens(messages: AgentMessage[]) {
  return estimateTranscriptTokens(convertToLlm(messages));
}

afterEach(() => vi.restoreAllMocks());

describe("system transcript helpers", () => {
  it("replays sections and tool removals without giving an unchanged prefix a new timestamp", () => {
    const previous = [initial, delta, assistant, user];
    const rebuilt = rebuildSystemTranscript(previous, [assistant, user]);
    expect(rebuilt).toBe(previous);
    expect(getCurrentSystemPrompt(rebuilt)).toBe(getCurrentSystemPrompt(previous));
    expect(getCurrentSystemMessage(rebuilt)?.sections).not.toHaveProperty("obsolete");
    expect(getCurrentTools(rebuilt)).toEqual([read]);
    expect(estimateContextTokens(rebuilt).usageTokens).toBe(1_000);
    expect(rebuildSystemTranscript(rebuilt, [assistant, user])[0]).toBe(rebuilt[0]);
  });

  it("keeps trailing system appends in place when the conversation grows", () => {
    const appended = appendSystemMessage([initial, assistant], "Reusable subagent sessions");
    const nextUser: AgentMessage = { role: "user", content: "next", timestamp: 3_000 };
    const rebuilt = rebuildSystemTranscript(appended, [assistant, nextUser]);
    expect(rebuilt[0]).toBe(initial);
    expect(rebuilt.at(-2)).toMatchObject({ role: "system", content: "Reusable subagent sessions" });
    expect(rebuilt.at(-1)).toBe(nextUser);
  });

  it("folds system state in front of a rewritten conversation without resurrecting predating usage", () => {
    const newer = { ...delta, timestamp: 3_000 };
    const rewritten = { ...assistant, content: [{ type: "text" as const, text: "Rewritten history" }] };
    const rebuilt = rebuildSystemTranscript([initial, assistant, newer], [rewritten]);
    expect(rebuilt[0]?.timestamp).toBe(3_000);
    expect(estimateContextTokens(rebuilt).lastUsageIndex).toBeNull();
  });

  it("appends dynamic system text without rewriting the head system message", () => {
    const messages = [initial, assistant];
    const appended = appendSystemMessage(messages, "Current reusable list");
    expect(appended[0]).toBe(initial);
    expect(appended[1]).toBe(assistant);
    expect(appended[2]).toMatchObject({ role: "system", content: "Current reusable list" });
    expect(appendSystemMessage(appended, "Current reusable list")).toBe(appended);
    expect(appendSystemMessage(messages, "")).toBe(messages);
  });


  it("preserves dynamic rows when unchanged base instructions are reapplied", () => {
    const messages = appendSystemMessage([initial, assistant], "Reusable session A");
    expect(replaceSystemPrompt(messages, "Base instructions")).toBe(messages);
    expect(systemPromptContent(messages)).toBe("Base instructions");
  });

  it("retains the current dynamic snapshot when base instructions really change", () => {
    const first = appendSystemMessage([initial, assistant], "Reusable session A");
    const latest = appendSystemMessage(first, "No reusable sessions");
    const changed = replaceSystemPrompt(latest, "New base instructions");
    expect(getCurrentSystemPrompt(changed)).toContain("No reusable sessions");
    expect(getCurrentSystemPrompt(changed)).not.toContain("Reusable session A");
    expect(getCurrentTools(changed)).toEqual([read, edit]);
  });

  it("keeps recovery nudges removable and only the latest reusable snapshot after compaction", () => {
    const first = appendSystemMessage([initial, assistant], "Reusable session A");
    const latest = appendSystemMessage(first, "No reusable sessions");
    const nudged = appendSystemMessage(latest, "One-shot recovery nudge", "transient");
    const nudge = nudged.at(-1);
    const compacted: AgentMessage = { role: "user", content: "Compacted history", timestamp: 9_000 };
    const rebuilt = rebuildSystemTranscript(nudged, [compacted]);
    expect(rebuilt).toContain(nudge);
    const cleaned = rebuilt.filter((message) => message !== nudge);
    expect(getCurrentSystemPrompt(cleaned)).not.toContain("One-shot recovery nudge");
    expect(getCurrentSystemPrompt(cleaned)).not.toContain("Reusable session A");
    expect(getCurrentSystemPrompt(cleaned)).toContain("No reusable sessions");
    expect(getCurrentTools(cleaned)).toEqual([read, edit]);
  });

  it("preserves positions when an unchanged checkpoint projection reallocates summary messages", () => {
    const summary: AgentMessage = { role: "user", content: "Same checkpoint summary", timestamp: 500 };
    const previous = [...appendSystemMessage([initial, summary], "Reusable session A"), assistant];
    const rebuilt = rebuildSystemTranscript(previous, [{ ...summary }, assistant, user]);
    expect(rebuilt.slice(0, previous.length)).toEqual(previous);
    expect(rebuilt[1]).toBe(summary);
    expect(rebuildSystemTranscript(rebuilt, [{ ...summary }, assistant, user])).toBe(rebuilt);
  });

  it("projects only owned dynamic rows as trailing context before provider system folding", () => {
    const previous = [initial, assistant];
    const appended = appendSystemMessage(previous, "Reusable session A");
    const projected = projectDynamicSystemMessages(appended);
    expect(projected.slice(0, previous.length)).toEqual(previous);
    expect(projected.at(-1)).toMatchObject({ role: "user", content: "<runtime-context>\nReusable session A\n</runtime-context>" });
    expect(getCurrentSystemPrompt(projected)).toBe(getCurrentSystemPrompt(previous));
    expect(getCurrentTools(projected)).toEqual(getCurrentTools(previous));
  });

  it("defers runtime context until all parallel tool results arrive", () => {
    const call: AssistantMessage = { ...assistant, content: [
      { type: "toolCall", id: "task-a", name: "Task", arguments: {} },
      { type: "toolCall", id: "task-b", name: "Task", arguments: {} },
    ] };
    const result = (id: string): AgentMessage => ({
      role: "toolResult", toolCallId: id, toolName: "Task", content: [{ type: "text", text: "done" }],
      isError: false, timestamp: 3_000,
    });
    const first = appendSystemMessage([initial, call], "Snapshot A");
    expect(projectDynamicSystemMessages(first)).toEqual([initial, call]);
    const resultA = result("task-a");
    const second = appendSystemMessage([...first, resultA], "Snapshot B");
    const resultB = result("task-b");
    const projected = projectDynamicSystemMessages([...second, resultB]);
    expect(projected.slice(0, 4)).toEqual([initial, call, resultA, resultB]);
    expect(projected.slice(4).map((message) => message.role)).toEqual(["user", "user"]);
    expect(projected[4]).toMatchObject({ content: "<runtime-context>\nSnapshot A\n</runtime-context>" });
    expect(projected[5]).toMatchObject({ content: "<runtime-context>\nSnapshot B\n</runtime-context>" });
  });

  it("keeps a complete recovery transcript and its deltas in place", () => {
    const previous = [initial, assistant, delta, user, { ...assistant, stopReason: "error" as const }];
    const retained = previous.slice(0, -1);
    expect(rebuildSystemTranscript(previous, retained)).toBe(retained);
    expect(syncSystemTools(retained, [read])).toBe(retained);
  });

  it("preserves object identity when setting either the same content or rendered prompt", () => {
    const messages = [initial, delta, assistant];
    expect(replaceSystemPrompt(messages, getCurrentSystemPrompt(messages))).toBe(messages);
    expect(replaceSystemPrompt(messages, systemPromptContent(messages))).toBe(messages);
  });

  it("invalidates usage only on a real content change and preserves structured sections and tools", () => {
    vi.spyOn(Date, "now").mockReturnValue(5_000);
    const changed = replaceSystemPrompt([initial, delta, assistant], "Replacement instructions");
    expect(changed[0]).toMatchObject({ content: "Replacement instructions", timestamp: 5_000, sections: { rules: "Updated rules" } });
    expect(getCurrentTools(changed)).toEqual([read]);
    expect(estimateContextTokens(changed).lastUsageIndex).toBeNull();
    expect(replaceSystemPrompt(changed, "Replacement instructions")).toBe(changed);
    expect(initial.content).toBe("Base instructions");
    const response = { ...assistant, timestamp: 6_000 };
    expect(estimateContextTokens(rebuildSystemTranscript(changed, [assistant, response])).usageTokens).toBe(1_000);
  });

  it("can clear content without clearing sections or tool declarations", () => {
    const changed = replaceSystemPrompt([initial, delta, assistant], "");
    expect(systemPromptContent(changed)).toBe("");
    expect(getCurrentSystemMessage(changed)?.sections).toEqual({ rules: "Updated rules" });
    expect(getCurrentTools(changed)).toEqual([read]);
  });

  it("restores a recovery prompt without flattening sections or backdating its response", () => {
    vi.spyOn(Date, "now").mockReturnValue(5_000);
    const messages = [initial, delta, assistant, user];
    const before = systemPromptContent(messages);
    const nudged = replaceSystemPrompt(messages, `${before}\n\nRecovery nudge`);
    const response = { ...assistant, timestamp: 6_000 };
    const restored = replaceSystemPrompt([...nudged, response], before);
    expect(getCurrentSystemPrompt(restored)).toBe(getCurrentSystemPrompt(messages));
    expect(getCurrentSystemMessage(restored)?.sections).toEqual({ rules: "Updated rules" });
    expect(restored[0]?.timestamp).toBeGreaterThan(response.timestamp);
    expect(estimateContextTokens(restored).usageTokens).toBe(0);
  });

  it("records tool additions, removals, and same-name schema replacements with fresh semantic time", () => {
    vi.spyOn(Date, "now").mockReturnValue(5_000);
    const replacement = { ...read, parameters: Type.Object({ file: Type.String() }) };
    const messages = [initial, assistant];
    const changed = syncSystemTools(messages, [replacement]);
    expect(changed[1]).toMatchObject({ timestamp: 5_000, toolsAdded: [replacement], toolsRemoved: [{ name: "Read" }, { name: "Edit" }] });
    expect(getCurrentTools(changed)).toEqual([replacement]);
    expect(estimateContextTokens(changed).usageTokens).toBe(0);
    expect(syncSystemTools(changed, [replacement])).toBe(changed);
    const response = { ...assistant, timestamp: 6_000 };
    expect(estimateContextTokens(rebuildSystemTranscript(changed, [assistant, response])).usageTokens).toBe(1_000);
    const cleared = syncSystemTools(changed, []);
    expect(getCurrentTools(rebuildSystemTranscript(cleared, [assistant]))).toEqual([]);
  });

  it("ignores executable-only tool changes instead of invalidating usage", () => {
    const messages = [initial, assistant];
    const executable = { ...read, label: "Read", execute: vi.fn() };
    expect(syncSystemTools(messages, [executable, edit])).toBe(messages);
    expect(estimateContextTokens(messages).usageTokens).toBe(1_000);
  });

  it("advances semantic time even if the clock shares a millisecond or moves backwards", () => {
    vi.spyOn(Date, "now").mockReturnValue(2_000);
    expect(replaceSystemPrompt([initial, assistant], "changed")[0]?.timestamp).toBe(2_001);
    vi.spyOn(Date, "now").mockReturnValue(500);
    expect(syncSystemTools([initial, assistant], [read])[1]?.timestamp).toBe(2_001);
  });

  it("initializes restored history conservatively without inventing an old timestamp", () => {
    vi.spyOn(Date, "now").mockReturnValue(5_000);
    const messages = initialSystemTranscript("Current config", [read], [assistant]);
    expect(messages[0]).toMatchObject({ content: "Current config", toolsAdded: [read], timestamp: 5_000 });
    expect(estimateContextTokens(messages).usageTokens).toBe(0);
    const rebuilt = rebuildSystemTranscript(messages, [assistant]);
    expect(rebuilt[0]).toBe(messages[0]);
    expect(estimateContextTokens(rebuilt).usageTokens).toBe(0);
  });

  it("preserves the empty transcript when there is no system state", () => {
    const messages: AgentMessage[] = [];
    expect(initialSystemTranscript("", [], messages)).toBe(messages);
    expect(rebuildSystemTranscript([], messages)).toBe(messages);
    expect(syncSystemTools(messages, [])).toBe(messages);
    expect(replaceSystemPrompt(messages, "")).toBe(messages);
  });

  it("does not trigger duplicate tool deltas in the real pi agent loop after a rebuild", async () => {
    const tool: AgentTool = {
      ...read, label: "Read", execute: async () => ({ content: [{ type: "text", text: "text" }], details: {} }),
    };
    const requests: AgentMessage[][] = [];
    const agent = new Agent({
      initialState: { tools: [tool], messages: [initial, delta, user] },
      streamFn: async (_model, context) => {
        requests.push(context.messages);
        const stream = createAssistantMessageEventStream();
        stream.push({ type: "done", reason: "stop", message: assistant });
        return stream;
      },
    });
    agent.state.messages = syncSystemTools(rebuildSystemTranscript(agent.state.messages, [user]), [tool]);
    const systems = agent.state.messages.filter((message) => message.role === "system");
    await agent.continue();
    expect(agent.state.errorMessage).toBeUndefined();
    expect(requests).toHaveLength(1);
    expect(requests[0]?.filter((message) => message.role === "system")).toEqual(systems);
    expect(getCurrentTools(requests[0]!)).toEqual([toToolDeclaration(tool)]);
    expect(agent.state.messages.filter((message) => message.role === "system")).toEqual(systems);
  });
});
