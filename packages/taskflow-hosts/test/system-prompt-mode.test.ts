import assert from "node:assert/strict";
import { test } from "node:test";
import type { AgentConfig, RunOptions, RunResult } from "taskflow-core";
import { runClaudeAgentTask } from "../src/claude-runner.ts";
import { runCodexAgentTask } from "../src/codex-runner.ts";
import { runGrokAgentTask } from "../src/grok-runner.ts";
import { runHermesAgentTask } from "../src/hermes-runner.ts";
import { runOpencodeAgentTask } from "../src/opencode-runner.ts";

type HostRunner = (
	defaultCwd: string,
	agents: AgentConfig[],
	agentName: string,
	task: string,
	opts: RunOptions,
	globalThinking?: string,
) => Promise<RunResult>;

const REPLACE_AGENT: AgentConfig = {
	name: "replace-worker",
	description: "replacement prompt worker",
	tools: ["read"],
	systemPrompt: "REPLACEMENT-PROMPT",
	systemPromptMode: "replace",
	source: "project",
	filePath: "/agents/replace-worker.md",
};

const UNSUPPORTED_HOSTS: ReadonlyArray<{
	host: string;
	binEnv: string;
	run: HostRunner;
}> = [
	{ host: "Codex", binEnv: "PI_TASKFLOW_CODEX_BIN", run: runCodexAgentTask },
	{ host: "Claude Code", binEnv: "PI_TASKFLOW_CLAUDE_BIN", run: runClaudeAgentTask },
	{ host: "OpenCode", binEnv: "PI_TASKFLOW_OPENCODE_BIN", run: runOpencodeAgentTask },
	{ host: "Grok Build", binEnv: "PI_TASKFLOW_GROK_BIN", run: runGrokAgentTask },
	{ host: "Hermes", binEnv: "PI_TASKFLOW_HERMES_BIN", run: runHermesAgentTask },
];

test("unsupported hosts reject systemPromptMode replace before process spawn", async () => {
	for (const { host, binEnv, run } of UNSUPPORTED_HOSTS) {
		const previous = process.env[binEnv];
		process.env[binEnv] = `/definitely/not/a/${host.toLowerCase().replaceAll(" ", "-")}/binary`;
		try {
			const result = await run("/tmp", [REPLACE_AGENT], REPLACE_AGENT.name, "inspect", { tools: ["read"] });
			const diagnostic = result.errorMessage ?? result.stderr;
			assert.equal(result.exitCode, 1, `${host} unexpectedly accepted replacement mode`);
			assert.match(diagnostic, /replace-worker/);
			assert.match(diagnostic, /systemPromptMode.*replace/i);
			assert.ok(diagnostic.includes(`active host '${host}'`), diagnostic);
			assert.match(diagnostic, /does not support replacement system prompts/i);
			assert.doesNotMatch(diagnostic, /ENOENT|spawn/i, `${host} must reject before the process seam`);
		} finally {
			if (previous === undefined) delete process.env[binEnv];
			else process.env[binEnv] = previous;
		}
	}
});
