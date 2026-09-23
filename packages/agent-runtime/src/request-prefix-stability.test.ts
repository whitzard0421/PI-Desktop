import { afterEach, describe, expect, it, vi } from "vitest";
import type { AgentEventEnvelope, SubagentDefinition, UiMessage } from "@pi-desktop/shared";
import {
  DesktopAgentRuntime,
  type RuntimeProviderConfig,
} from "./runtime.js";

const subagentRuns = vi.hoisted(() => ({
  deferred: false,
  calls: [] as Array<{
    parentToolCallId?: string;
    task?: string;
    initialMessages?: unknown;
  }>,
  instances: [] as Array<{ resolve: (result: unknown) => void; settled: boolean }>,
  resolveRun: undefined as ((result: unknown) => void) | undefined,
}));

vi.mock("./subagent.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./subagent.js")>();
  return {
    ...actual,
    SubagentRun: class {
      constructor(options: {
        definition: { name: string };
        sessionId: string;
        turnId: string;
        parentToolCallId: string;
        task: string;
        initialMessages?: unknown;
        onEvent: (envelope: AgentEventEnvelope) => void;
        signal?: AbortSignal;
      }) {
        subagentRuns.calls.push({
          parentToolCallId: options.parentToolCallId,
          task: options.task,
          initialMessages: options.initialMessages,
        });
        // A resumed Task needs at least one chain row in transcriptHistory.
        options.onEvent({
          sessionId: options.sessionId,
          turnId: options.turnId,
          ts: Date.now(),
          parentToolCallId: options.parentToolCallId,
          agentName: options.definition.name,
          event: {
            type: "message_end",
            message: {
              id: `child-${options.parentToolCallId}`,
              role: "assistant",
              content: "scanned the workspace",
              createdAt: new Date().toISOString(),
              status: "complete",
            },
          },
        } as AgentEventEnvelope);
        this.signal = options.signal;
      }

      private signal?: AbortSignal;

      run() {
        if (!subagentRuns.deferred) {
          return Promise.resolve({
            agentName: "explorer",
            status: "completed",
            report: "done",
            turns: 1,
            toolCalls: 0,
          });
        }
        return new Promise((resolve) => {
          const instance = { resolve, settled: false };
          subagentRuns.instances.push(instance);
          subagentRuns.resolveRun = (result: unknown) => {
            const next = subagentRuns.instances.find((entry) => !entry.settled);
            if (next) {
              next.settled = true;
              next.resolve(result);
            }
          };
          this.signal?.addEventListener(
            "abort",
            () => {
              if (instance.settled) return;
              instance.settled = true;
              resolve({
                agentName: "explorer",
                status: "aborted",
                report: "The delegated task was aborted.",
                turns: 0,
                toolCalls: 0,
              });
            },
            { once: true },
          );
        });
      }
    },
  };
});

const explorer: SubagentDefinition = {
  name: "explorer",
  description: "Search the workspace and report findings.",
  tools: ["Read", "Glob", "Grep"],
  prompt: "Report file paths and line numbers.",
  source: "builtin",
};

const responsesProvider: RuntimeProviderConfig = {
  id: "responses",
  name: "Responses",
  baseUrl: "http://127.0.0.1:9/v1",
  modelId: "responses-test",
  apiKey: "sk-test",
  authKind: "api_key",
  apiStyle: "responses",
  supportsReasoning: false,
  supportedThinkingLevels: ["off"],
  modelConfig: {
    source: "generic",
    name: "Responses test",
    baseUrl: "http://127.0.0.1:9/v1",
    reasoning: false,
    input: ["text"],
    contextWindow: 128_000,
    maxTokens: 1_024,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  },
};

const runtimeFixtures = new Set<DesktopAgentRuntime>();
const pendingPrompts = new Set<Promise<unknown>>();
const runtimeEvents: AgentEventEnvelope[] = [];

type CapturedBody = Record<string, unknown>;
type SseReply = string | ((body: CapturedBody) => string);

function sseEvent(type: string, payload: Record<string, unknown>): string {
  return `event: ${type}\ndata: ${JSON.stringify({ type, ...payload })}\n\n`;
}

function usage() {
  return {
    input_tokens: 20,
    input_tokens_details: { cached_tokens: 0 },
    output_tokens: 4,
    output_tokens_details: { reasoning_tokens: 0 },
    total_tokens: 24,
  };
}

function completedSse(text: string, serial = "done"): string {
  const response = {
    object: "response",
    id: `resp_${serial}`,
    created_at: 1,
    model: "responses-test",
    output: [
      {
        id: `msg_${serial}`,
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text }],
        status: "completed",
      },
    ],
    status: "completed",
    usage: usage(),
  };
  return [
    sseEvent("response.created", { sequence_number: 0, response: { id: response.id } }),
    sseEvent("response.output_item.added", { sequence_number: 1, output_index: 0, item: { ...response.output[0], content: [] } }),
    sseEvent("response.output_text.delta", {
      sequence_number: 2,
      output_index: 0,
      content_index: 0,
      delta: text,
    }),
    sseEvent("response.output_item.done", { sequence_number: 3, output_index: 0, item: response.output[0] }),
    sseEvent("response.completed", { sequence_number: 4, response }),
  ].join("");
}

function toolCallSse(name: string, args: Record<string, unknown>, serial: string): string {
  const argsJson = JSON.stringify(args);
  const item = {
    id: `fc_${serial}`,
    call_id: `call_${serial}`,
    type: "function_call",
    name,
    arguments: "",
    status: "in_progress",
  };
  const completed = { ...item, status: "completed", arguments: argsJson };
  const locator = { output_index: 0, item_id: item.id };
  const response = {
    object: "response",
    id: `resp_${serial}`,
    created_at: 1,
    model: "responses-test",
    output: [completed],
    status: "completed",
    usage: usage(),
  };
  return [
    sseEvent("response.created", { sequence_number: 0, response: { id: response.id } }),
    sseEvent("response.output_item.added", { sequence_number: 1, output_index: 0, item }),
    sseEvent("response.function_call_arguments.delta", {
      sequence_number: 2,
      ...locator,
      delta: argsJson,
    }),
    sseEvent("response.function_call_arguments.done", {
      sequence_number: 3,
      ...locator,
      arguments: argsJson,
    }),
    sseEvent("response.output_item.done", { sequence_number: 4, output_index: 0, item: completed }),
    sseEvent("response.completed", { sequence_number: 5, response }),
  ].join("");
}

function createRuntime(
  overrides: {
    history?: UiMessage[];
    subagents?: SubagentDefinition[];
  } = {},
) {
  const runtime = new DesktopAgentRuntime({
    host: { call: vi.fn(async () => undefined), onNotification: vi.fn(() => () => {}) } as never,
    sessionId: "session-prefix",
    mode: "agent",
    provider: responsesProvider,
    commandShell: {
      id: "bash",
      label: "Bash",
      dialect: "posix",
      available: true,
      isDefault: true,
    },
    thinkingLevel: "off",
    infiniteProviderRetry: false,
    history: overrides.history,
    subagents: overrides.subagents,
    compactionSettings: { enabled: false } as never,
    onEvent: (envelope) => { runtimeEvents.push(envelope); },
  });
  runtimeFixtures.add(runtime);
  const prompt = runtime.prompt.bind(runtime);
  runtime.prompt = (...args) => {
    const pending = prompt(...args);
    pendingPrompts.add(pending);
    void pending.then(() => pendingPrompts.delete(pending), () => pendingPrompts.delete(pending));
    return pending;
  };
  return runtime;
}

function toolNames(body: CapturedBody): string[] {
  const tools = Array.isArray(body.tools) ? body.tools : [];
  return tools
    .map((tool) => {
      if (!tool || typeof tool !== "object") return "";
      const record = tool as { name?: unknown; function?: { name?: unknown } };
      if (typeof record.name === "string") return record.name;
      if (typeof record.function?.name === "string") return record.function.name;
      return "";
    })
    .filter(Boolean);
}

function requestBlobs(body: CapturedBody): string[] {
  const blobs: string[] = [];
  if (typeof body.instructions === "string") blobs.push(body.instructions);
  const input = Array.isArray(body.input) ? body.input : [];
  for (const item of input) {
    if (!item || typeof item !== "object") continue;
    const record = item as { content?: unknown; output?: unknown };
    if (typeof record.content === "string") blobs.push(record.content);
    if (Array.isArray(record.content)) {
      for (const part of record.content) {
        if (part && typeof part === "object" && typeof part.text === "string") blobs.push(part.text);
      }
    }
    if (typeof record.output === "string") blobs.push(record.output);
  }
  return blobs;
}

function latestAvailability(body: CapturedBody): string {
  let latest = "";
  for (const blob of requestBlobs(body)) {
    if (
      blob.includes("Reusable subagent sessions") ||
      blob.includes("No reusable subagent sessions")
    ) {
      latest = blob;
    }
  }
  return latest;
}

function extractDelegationId(body: CapturedBody): string {
  let found = "";
  for (const blob of requestBlobs(body)) {
    const match = blob.match(
      /Delegation ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}) started/,
    );
    if (match) found = match[1]!;
  }
  if (!found) throw new Error("delegation id missing from serialized Responses input");
  return found;
}

function assertUnchangedPrefix(earlier: CapturedBody, later: CapturedBody): void {
  expect(later.instructions).toEqual(earlier.instructions);
  const earlierInput = Array.isArray(earlier.input) ? earlier.input : [];
  const laterInput = Array.isArray(later.input) ? later.input : [];
  expect(laterInput.slice(0, earlierInput.length)).toEqual(earlierInput);
  expect(later).not.toHaveProperty("cached_tokens");
}

function assertToolResults(body: CapturedBody): void {
  const input = (Array.isArray(body.input) ? body.input : []) as Array<Record<string, unknown>>;
  const calls = input.filter((item) => item.type === "function_call");
  const results = input.filter((item) => item.type === "function_call_output");
  for (const call of calls) {
    expect(results.filter((item) => item.call_id === call.call_id), String(call.call_id)).toHaveLength(1);
  }
  for (const result of results) {
    expect(calls.filter((item) => item.call_id === result.call_id), String(result.call_id)).toHaveLength(1);
    expect(JSON.stringify(result.output)).not.toContain("No result provided");
  }
}

function assertExactTools(earlier: CapturedBody, later: CapturedBody): void {
  expect(later.tools).toEqual(earlier.tools);
  assertToolResults(earlier);
  assertToolResults(later);
}

function settleExplorer(report: string): void {
  subagentRuns.resolveRun?.({
    agentName: "explorer",
    status: "completed",
    report,
    turns: 1,
    toolCalls: 0,
  });
}

describe("request prefix stability (issue #913)", () => {
  const captured: CapturedBody[] = [];

  afterEach(async () => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      await Promise.race([
        Promise.all([...runtimeFixtures].map((runtime) => runtime.dispose()))
          .then(() => Promise.allSettled([...pendingPrompts])),
        new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("Fixture cleanup timed out")), 2_000); }),
      ]);
    } finally {
      clearTimeout(timer);
      runtimeFixtures.clear();
      pendingPrompts.clear();
    }
    runtimeEvents.length = 0;
    captured.length = 0;
    subagentRuns.deferred = false;
    subagentRuns.calls.length = 0;
    subagentRuns.instances.length = 0;
    subagentRuns.resolveRun = undefined;
  });

  function interceptProviderFetch(runtime: DesktopAgentRuntime, replies: SseReply[]): void {
    const capturingFetch = async (_input: RequestInfo | URL, init?: RequestInit) => {
      const raw = typeof init?.body === "string" ? init.body : "{}";
      const body = JSON.parse(raw) as CapturedBody;
      captured.push(body);
      const next = replies.shift();
      if (next === undefined) throw new Error("Unexpected provider request in prefix fixture");
      const sse = typeof next === "function" ? next(body) : next;
      return new Response(sse, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      });
    };
    const models = (runtime as any).models;
    for (const method of ["streamSimple", "stream"] as const) {
      const original = models[method].bind(models);
      models[method] = (model: unknown, context: unknown, options: Record<string, unknown> = {}) =>
        original(model, context, { ...options, fetch: capturingFetch, apiKey: "sk-test" });
    }
  }

  async function waitCaptured(count: number): Promise<void> {
    await vi.waitFor(() => {
      expect(captured.length).toBeGreaterThanOrEqual(count);
    }, { timeout: 10_000 });
  }

  it("keeps Responses tools JSON and input prefix across ToolSearch, Task auto-delivery, and resume", async () => {
    subagentRuns.deferred = true;
    const runtime = createRuntime({ subagents: [explorer] });
    const deferredNames = [...((runtime as any).deferredToolNames as Set<string>)];
    expect(deferredNames).toContain("BrowserPreview");
    expect(deferredNames.length).toBeGreaterThan(1);

    interceptProviderFetch(runtime, [
      toolCallSse("ToolSearch", { query: "BrowserPreview" }, "search1"),
      toolCallSse(
        "Task",
        { agent: "explorer", task: "Find the bug.", description: "find the bug" },
        "task1",
      ),
      completedSse("continuing while explorer runs", "run1"),
      completedSse("integrated the first report", "deliver1"),
      completedSse("still unused, next step", "user2"),
      () => {
        const resume = extractDelegationId(captured[2]!);
        return toolCallSse(
          "Task",
          {
            agent: "explorer",
            task: "Continue from the last scan.",
            description: "continue explorer",
            resume,
          },
          "task2",
        );
      },
      completedSse("continuing while resume runs", "run2"),
      completedSse("integrated the resumed report", "deliver2"),
    ]);

    const firstTurn = runtime.prompt(
      "Activate BrowserPreview without using it, then start explorer.",
      "user-1",
      "turn-1",
    );
    await waitCaptured(3);

    expect(toolNames(captured[0]!)).toContain("ToolSearch");
    expect(toolNames(captured[0]!)).not.toContain("BrowserPreview");
    expect(toolNames(captured[1]!), JSON.stringify(runtimeEvents.slice(-5))).toContain("BrowserPreview");
    expect(toolNames(captured[1]!).filter((name) => deferredNames.includes(name))).toEqual([
      "BrowserPreview",
    ]);
    expect(toolNames(captured[1]!)).not.toContain("Glob");
    expect(requestBlobs(captured[1]!).some((blob) => blob.includes("Activated on-demand tools"))).toBe(
      true,
    );

    const firstDelegationId = extractDelegationId(captured[2]!);
    expect(latestAvailability(captured[2]!)).not.toContain(firstDelegationId);
    assertUnchangedPrefix(captured[1]!, captured[2]!);
    assertExactTools(captured[1]!, captured[2]!);

    settleExplorer("src/app.ts:12 misses the null check.");
    await firstTurn;
    expect(captured).toHaveLength(4);
    expect(requestBlobs(captured[3]!).join("\n")).toContain("Integrate their reports");
    expect(requestBlobs(captured[3]!).join("\n")).toContain("src/app.ts:12 misses the null check.");
    expect(latestAvailability(captured[3]!)).toContain(firstDelegationId);
    assertUnchangedPrefix(captured[2]!, captured[3]!);
    assertExactTools(captured[2]!, captured[3]!);

    await runtime.prompt("Still unused. Next step.", "user-2", "turn-2");
    expect(captured).toHaveLength(5);
    expect(toolNames(captured[4]!)).toContain("BrowserPreview");
    expect(toolNames(captured[4]!).filter((name) => deferredNames.includes(name))).toEqual([
      "BrowserPreview",
    ]);
    assertUnchangedPrefix(captured[3]!, captured[4]!);
    assertExactTools(captured[3]!, captured[4]!);
    expect(latestAvailability(captured[4]!)).toContain(firstDelegationId);

    const resumeTurn = runtime.prompt("Resume the explorer.", "user-3", "turn-3");
    await waitCaptured(7);
    expect(subagentRuns.calls).toHaveLength(2);
    expect(subagentRuns.calls[1]?.initialMessages).toBeDefined();
    const resumeDelegationId = extractDelegationId(captured[6]!);
    expect(resumeDelegationId).not.toBe(firstDelegationId);
    expect(latestAvailability(captured[6]!)).not.toContain(resumeDelegationId);
    expect(latestAvailability(captured[6]!)).not.toMatch(
      new RegExp(`${resumeDelegationId}|${firstDelegationId}`),
    );
    assertUnchangedPrefix(captured[5]!, captured[6]!);
    assertUnchangedPrefix(captured[4]!, captured[5]!);
    assertExactTools(captured[4]!, captured[5]!);
    assertExactTools(captured[5]!, captured[6]!);

    settleExplorer("lexer coverage is next.");
    await resumeTurn;
    expect(captured).toHaveLength(8);
    expect(requestBlobs(captured[7]!).join("\n")).toContain("lexer coverage is next.");
    expect(latestAvailability(captured[7]!)).toContain(resumeDelegationId);
    assertUnchangedPrefix(captured[6]!, captured[7]!);
    assertExactTools(captured[6]!, captured[7]!);

    await runtime.dispose();
  }, 20_000);

  it("keeps one result per call when a child settles during TaskWait", async () => {
    subagentRuns.deferred = true;
    const runtime = createRuntime({ subagents: [explorer] });
    interceptProviderFetch(runtime, [
      toolCallSse("Task", { agent: "explorer", task: "Inspect the file.", description: "inspect file" }, "wait-task"),
      (body) => toolCallSse("TaskWait", { delegationIds: [extractDelegationId(body)], mode: "all", timeoutSeconds: 10 }, "wait-call"),
      completedSse("The waited report is integrated.", "wait-done"),
    ]);
    const turn = runtime.prompt("Delegate the inspection and wait for its report.", "wait-user", "wait-turn");
    await vi.waitFor(() => {
      expect(runtimeEvents.some(({ event }) => event.type === "tool_start" && event.toolName === "TaskWait")).toBe(true);
    });
    const delegationId = extractDelegationId(captured[1]!);
    settleExplorer("The waited inspection found a missing guard.");
    await turn;
    expect(captured).toHaveLength(3);
    expect(requestBlobs(captured[2]!).join("\n")).toContain("The waited inspection found a missing guard.");
    expect(latestAvailability(captured[2]!)).toContain(delegationId);
    assertUnchangedPrefix(captured[1]!, captured[2]!);
    assertExactTools(captured[1]!, captured[2]!);
  });

  it("keeps the serialized Responses prefix after restoring an unused details.activated ToolSearch", async () => {
    const runtime = createRuntime({
      history: [
        {
          id: "user-1",
          role: "user",
          content: "Preview later.",
          createdAt: "2026-09-01T00:00:00.000Z",
          status: "complete",
        },
        {
          id: "assistant-1",
          role: "assistant",
          content: "I will load BrowserPreview.",
          createdAt: "2026-09-01T00:00:01.000Z",
          status: "complete",
        },
        {
          id: "tool-search-1",
          role: "tool",
          content: "",
          createdAt: "2026-09-01T00:00:02.000Z",
          status: "complete",
          toolName: "ToolSearch",
          toolCallId: "call-search-1",
          toolStatus: "success",
          toolArgs: { query: "BrowserPreview" },
          toolResult: {
            content: [{ type: "text", text: "Activated on-demand tools: BrowserPreview." }],
            details: {
              query: "BrowserPreview",
              activated: ["BrowserPreview"],
            },
          },
        },
      ],
    });
    interceptProviderFetch(runtime, [
      completedSse("first", "restore1"),
      completedSse("second", "restore2"),
    ]);

    await runtime.prompt("Continue without using the preview yet.", "user-2", "turn-2");
    await runtime.prompt("Still unused. Next step.", "user-3", "turn-3");

    expect(captured).toHaveLength(2);
    expect(toolNames(captured[0]!)).toContain("BrowserPreview");
    expect(toolNames(captured[1]!)).toContain("BrowserPreview");
    assertExactTools(captured[0]!, captured[1]!);
    assertUnchangedPrefix(captured[0]!, captured[1]!);
    await runtime.dispose();
  });
});
