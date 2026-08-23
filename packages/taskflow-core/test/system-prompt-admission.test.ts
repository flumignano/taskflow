import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { test } from "node:test";
import type { AgentConfig } from "../src/agents.ts";
import { CacheStore } from "../src/cache.ts";
import { forkRunForResume } from "../src/resume.ts";
import { executeTaskflow, recomputeTaskflow, type RuntimeDeps } from "../src/runtime.ts";
import type { SystemPromptMode } from "../src/host/runner-types.ts";
import type { Taskflow } from "../src/schema.ts";
import type { RunState } from "../src/store.ts";
import { emptyUsage } from "../src/usage.ts";

function agent(name: string, systemPromptMode?: SystemPromptMode): AgentConfig {
	return {
		name,
		description: `${name} agent`,
		systemPrompt: `${name} prompt`,
		...(systemPromptMode === undefined ? {} : { systemPromptMode }),
		source: "project",
		filePath: `/agents/${name}.md`,
	};
}

function state(def: Taskflow, cwd: string, runId: string): RunState {
	return {
		runId,
		flowName: def.name,
		def,
		args: {},
		status: "running",
		phases: {},
		createdAt: Date.now(),
		updatedAt: Date.now(),
		cwd,
	};
}

function tempDir(): string {
	return fs.mkdtempSync(path.join(os.tmpdir(), "tf-prompt-admission-"));
}

function deps(
	cwd: string,
	agents: AgentConfig[],
	calls: string[],
	systemPromptModes: readonly SystemPromptMode[] | undefined,
	extra: Partial<RuntimeDeps> = {},
): RuntimeDeps {
	return {
		cwd,
		agents,
		systemPromptModes,
		runTask: async (_cwd, _agents, agentName, task) => {
			calls.push(`${agentName}:${task}`);
			return {
				agent: agentName,
				task,
				exitCode: 0,
				output: task === "legacy" ? "legacy-append-output" : `out:${task}`,
				stderr: "",
				usage: emptyUsage(),
				stopReason: "end",
			};
		},
		...extra,
	};
}

test("unsupported replace fails before a matching cross-run cache result is accepted", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const cacheStore = new CacheStore(cwd);
		const replaceAgent = agent("worker", "replace");
		const def: Taskflow = {
			name: "replace-cross-run",
			phases: [
				{ id: "work", agent: "worker", task: "legacy" },
				{ id: "consume", agent: "worker", task: "consume {steps.work.output}", dependsOn: ["work"], final: true },
			],
		};
		const shared = { cacheStore, cacheScopeDefault: "cross-run" as const };
		const primed = await executeTaskflow(
			state(def, cwd, "replace-cache-prime"),
			deps(cwd, [replaceAgent], calls, ["append", "replace"], shared),
		);
		assert.equal(primed.ok, true);
		assert.equal(calls.length, 2);

		const rejected = await executeTaskflow(
			state(def, cwd, "replace-cache-reject"),
			deps(cwd, [replaceAgent], calls, ["append"], shared),
		);
		assert.equal(rejected.ok, false);
		assert.equal(calls.length, 2, "unsupported host must neither spawn nor run downstream on cached legacy output");
		assert.doesNotMatch(rejected.finalOutput, /^out:consume legacy-append-output$/);
		assert.equal(rejected.state.phases.work?.status, "failed");
		assert.equal(rejected.state.phases.work?.output, undefined);
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("unsupported replace fails before ordinary resume accepts copied completed output", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const replaceAgent = agent("worker", "replace");
		const def: Taskflow = {
			name: "replace-resume",
			phases: [{ id: "work", agent: "worker", task: "legacy", final: true }],
		};
		const parentResult = await executeTaskflow(
			state(def, cwd, "replace-resume-parent"),
			deps(cwd, [replaceAgent], calls, ["append", "replace"]),
		);
		assert.equal(parentResult.ok, true);
		parentResult.state.status = "paused"; // ordinary resume accepts paused parents and copies done phases
		const parentSnapshot = structuredClone(parentResult.state);
		const child = forkRunForResume(parentResult.state, { cwd });

		const rejected = await executeTaskflow(child, deps(cwd, [replaceAgent], calls, ["append"]));
		assert.equal(rejected.ok, false);
		assert.equal(calls.length, 1, "ordinary resume must fail admission without spawning");
		assert.notEqual(rejected.finalOutput, "legacy-append-output");
		assert.equal(rejected.state.phases.work?.status, "failed");
		assert.equal(rejected.state.phases.work?.output, undefined);
		assert.deepEqual(parentResult.state, parentSnapshot, "resume admission must not mutate the parent run");
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("unsupported replace direct execution rejects before spawn", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const def: Taskflow = {
			name: "replace-direct",
			phases: [{ id: "work", agent: "worker", task: "legacy", final: true }],
		};
		const result = await executeTaskflow(
			state(def, cwd, "replace-direct"),
			deps(cwd, [agent("worker", "replace")], calls, ["append"]),
		);
		assert.equal(result.ok, false);
		assert.equal(calls.length, 0);
		assert.match(result.finalOutput, /systemPromptMode.*replace.*active host/i);
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("missing host capability fails closed for explicit replace", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const def: Taskflow = {
			name: "replace-unknown-host",
			phases: [{ id: "work", agent: "worker", task: "legacy", final: true }],
		};
		const result = await executeTaskflow(
			state(def, cwd, "replace-unknown-host"),
			deps(cwd, [agent("worker", "replace")], calls, undefined),
		);
		assert.equal(result.ok, false);
		assert.equal(calls.length, 0);
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("unsupported replace fails recompute admission before prior state is reused or re-executed", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const replaceAgent = agent("worker", "replace");
		const def: Taskflow = {
			name: "replace-recompute",
			phases: [{ id: "work", agent: "worker", task: "legacy", final: true }],
		};
		const parent = await executeTaskflow(
			state(def, cwd, "replace-recompute-parent"),
			deps(cwd, [replaceAgent], calls, ["append", "replace"]),
		);
		const parentSnapshot = structuredClone(parent.state);
		await assert.rejects(
			recomputeTaskflow(
				parent.state,
				deps(cwd, [replaceAgent], calls, ["append"]),
				["work"],
				{ dryRun: false },
			),
			/host capability admission.*systemPromptMode.*replace/i,
		);
		assert.equal(calls.length, 1);
		assert.deepEqual(parent.state, parentSnapshot, "recompute admission must not mutate the source run");
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("append and omitted modes preserve unsupported-host cache and resume reuse", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const cacheStore = new CacheStore(cwd);
		const appendDef: Taskflow = {
			name: "append-cache",
			phases: [{ id: "work", agent: "append-worker", task: "append", final: true }],
		};
		const shared = { cacheStore, cacheScopeDefault: "cross-run" as const };
		await executeTaskflow(
			state(appendDef, cwd, "append-cache-1"),
			deps(cwd, [agent("append-worker", "append")], calls, ["append"], shared),
		);
		const cached = await executeTaskflow(
			state(appendDef, cwd, "append-cache-2"),
			deps(cwd, [agent("append-worker", "append")], calls, ["append"], shared),
		);
		assert.equal(cached.ok, true);
		assert.equal(cached.state.phases.work.cacheHit, "cross-run");

		const defaultDef: Taskflow = {
			name: "default-resume",
			phases: [{ id: "work", agent: "default-worker", task: "default", final: true }],
		};
		const parent = await executeTaskflow(
			state(defaultDef, cwd, "default-resume-parent"),
			deps(cwd, [agent("default-worker")], calls, ["append"]),
		);
		parent.state.status = "paused";
		const resumed = await executeTaskflow(
			forkRunForResume(parent.state, { cwd }),
			deps(cwd, [agent("default-worker")], calls, ["append"]),
		);
		assert.equal(resumed.ok, true);
		assert.equal(resumed.state.phases.work.cacheHit, "run-only");
		assert.equal(calls.length, 2, "one append execution and one default execution only");
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("Pi-supported replace remains admissible through normal execution and resume", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const replaceAgent = agent("worker", "replace");
		const def: Taskflow = {
			name: "pi-replace",
			phases: [{ id: "work", agent: "worker", task: "replace", final: true }],
		};
		const parent = await executeTaskflow(
			state(def, cwd, "pi-replace-parent"),
			deps(cwd, [replaceAgent], calls, ["append", "replace"]),
		);
		assert.equal(parent.ok, true);
		parent.state.status = "paused";
		const resumed = await executeTaskflow(
			forkRunForResume(parent.state, { cwd }),
			deps(cwd, [replaceAgent], calls, ["append", "replace"]),
		);
		assert.equal(resumed.ok, true);
		assert.equal(resumed.state.phases.work.cacheHit, "run-only");
		assert.equal(calls.length, 1);
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("an unused replace agent does not block a flow that references an append agent", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const def: Taskflow = {
			name: "unused-replace",
			phases: [{ id: "work", agent: "used", task: "append", final: true }],
		};
		const result = await executeTaskflow(
			state(def, cwd, "unused-replace"),
			deps(cwd, [agent("unused", "replace"), agent("used", "append")], calls, ["append"]),
		);
		assert.equal(result.ok, true);
		assert.deepEqual(calls, ["used:append"]);
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("unsupported replace in a saved nested flow fails before the parent flow cache is reused", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const cacheStore = new CacheStore(cwd);
		const child: Taskflow = {
			name: "replace-child",
			phases: [{ id: "child-work", agent: "worker", task: "legacy", final: true }],
		};
		const parent: Taskflow = {
			name: "replace-parent",
			phases: [{ id: "nested", type: "flow", use: child.name, final: true }],
		};
		const shared = {
			cacheStore,
			cacheScopeDefault: "cross-run" as const,
			loadFlow: (name: string) => name === child.name ? child : undefined,
		};
		const primed = await executeTaskflow(
			state(parent, cwd, "nested-prime"),
			deps(cwd, [agent("worker", "replace")], calls, ["append", "replace"], shared),
		);
		assert.equal(primed.ok, true);
		const rejected = await executeTaskflow(
			state(parent, cwd, "nested-reject"),
			deps(cwd, [agent("worker", "replace")], calls, ["append"], shared),
		);
		assert.equal(rejected.ok, false);
		assert.equal(calls.length, 1);
		assert.equal(rejected.state.phases.nested?.status, "failed");
		assert.notEqual(rejected.finalOutput, "legacy-append-output");
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("event-kernel nested execution preserves unsupported replace admission", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const child: Taskflow = {
			name: "kernel-replace-child",
			phases: [{ id: "child-work", agent: "worker", task: "legacy", final: true }],
		};
		const parent: Taskflow = {
			name: "kernel-replace-parent",
			phases: [{ id: "nested", type: "flow", def: child, final: true }],
		};
		const result = await executeTaskflow(
			state(parent, cwd, "kernel-replace"),
			deps(cwd, [agent("worker", "replace")], calls, ["append"], { eventKernel: true }),
		);
		assert.equal(result.ok, false);
		assert.equal(calls.length, 0);
		assert.match(
			[result.finalOutput, ...Object.values(result.state.phases).map((phase) => phase.error ?? "")].join("\n"),
			/systemPromptMode.*replace.*active host/i,
		);
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});
