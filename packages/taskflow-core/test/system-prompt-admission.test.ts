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
				output: task.includes("## Target under evaluation")
					? '{"score":1,"verdict":"pass","reason":"test judge pass"}'
					: task === "legacy" ? "legacy-append-output" : `out:${task}`,
				stderr: "",
				usage: emptyUsage(),
				stopReason: "end",
			};
		},
		...extra,
	};
}

function scoringGate(
	id: string,
	judgeAgent: string | undefined,
	phaseAgent = "append-worker",
): Taskflow["phases"][number] {
	return {
		id,
		type: "gate",
		agent: phaseAgent,
		score: {
			target: "payload",
			scorers: [{ type: "contains", value: "not-present" }],
			judge: { ...(judgeAgent === undefined ? {} : { agent: judgeAgent }), task: "judge payload" },
		},
		final: true,
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


test("unsupported replacement scoring judge fails before cross-run cache trust", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const cacheStore = new CacheStore(cwd);
		const def: Taskflow = {
			name: "replace-score-cache",
			phases: [
				{ id: "seed", agent: "append-worker", task: "legacy", cache: { scope: "cross-run" } },
				{ ...scoringGate("quality", "replace-judge"), dependsOn: ["seed"] },
			],
		};
		const agents = [agent("append-worker", "append"), agent("replace-judge", "replace")];
		const shared = { cacheStore };
		const primed = await executeTaskflow(
			state(def, cwd, "replace-score-cache-prime"),
			deps(cwd, agents, calls, ["append", "replace"], shared),
		);
		assert.equal(primed.ok, true);
		assert.equal(calls.length, 2);

		const rejected = await executeTaskflow(
			state(def, cwd, "replace-score-cache-reject"),
			deps(cwd, agents, calls, ["append"], shared),
		);
		assert.equal(rejected.ok, false);
		assert.equal(calls.length, 2, "admission must reject before cache lookup or judge spawn");
		assert.equal(rejected.state.phases.seed, undefined);
		assert.equal(rejected.state.phases.quality?.status, "failed");
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("unsupported replacement scoring judge fails before ordinary resume trust", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const agents = [agent("append-worker", "append"), agent("replace-judge", "replace")];
		const def: Taskflow = { name: "replace-score-resume", phases: [scoringGate("quality", "replace-judge")] };
		const parent = await executeTaskflow(
			state(def, cwd, "replace-score-resume-parent"),
			deps(cwd, agents, calls, ["append", "replace"]),
		);
		assert.equal(parent.ok, true);
		parent.state.status = "paused";
		const parentSnapshot = structuredClone(parent.state);

		const rejected = await executeTaskflow(
			forkRunForResume(parent.state, { cwd }),
			deps(cwd, agents, calls, ["append"]),
		);
		assert.equal(rejected.ok, false);
		assert.equal(calls.length, 1, "admission must reject without respawning or accepting run-only state");
		assert.notEqual(rejected.state.phases.quality?.cacheHit, "run-only");
		assert.deepEqual(parent.state, parentSnapshot);
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("unsupported replacement scoring judge fails recompute admission", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const agents = [agent("append-worker", "append"), agent("replace-judge", "replace")];
		const def: Taskflow = { name: "replace-score-recompute", phases: [scoringGate("quality", "replace-judge")] };
		const parent = await executeTaskflow(
			state(def, cwd, "replace-score-recompute-parent"),
			deps(cwd, agents, calls, ["append", "replace"]),
		);
		assert.equal(parent.ok, true);
		const parentSnapshot = structuredClone(parent.state);

		await assert.rejects(
			recomputeTaskflow(
				parent.state,
				deps(cwd, agents, calls, ["append"]),
				["quality"],
				{ dryRun: false },
			),
			/host capability admission.*systemPromptMode.*replace/i,
		);
		assert.equal(calls.length, 1, "recompute must reject before reuse or execution");
		assert.deepEqual(parent.state, parentSnapshot);
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("nested parent cache cannot hide an unsupported replacement scoring judge", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const cacheStore = new CacheStore(cwd);
		const child: Taskflow = { name: "replace-score-child", phases: [scoringGate("quality", "replace-judge")] };
		const parent: Taskflow = {
			name: "replace-score-parent",
			phases: [{ id: "nested", type: "flow", use: child.name, cache: { scope: "cross-run" }, final: true }],
		};
		const agents = [agent("append-worker", "append"), agent("replace-judge", "replace")];
		const shared = {
			cacheStore,
			loadFlow: (name: string) => name === child.name ? child : undefined,
		};
		const primed = await executeTaskflow(
			state(parent, cwd, "replace-score-parent-prime"),
			deps(cwd, agents, calls, ["append", "replace"], shared),
		);
		assert.equal(primed.ok, true);
		assert.equal(calls.length, 1);

		const rejected = await executeTaskflow(
			state(parent, cwd, "replace-score-parent-reject"),
			deps(cwd, agents, calls, ["append"], shared),
		);
		assert.equal(rejected.ok, false);
		assert.equal(calls.length, 1);
		assert.equal(rejected.state.phases.nested?.status, "failed");
		assert.notEqual(rejected.state.phases.nested?.cacheHit, "cross-run");
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("Pi-supported replacement scoring judge remains admissible", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const agents = [agent("append-worker", "append"), agent("replace-judge", "replace")];
		const def: Taskflow = { name: "pi-replace-score", phases: [scoringGate("quality", "replace-judge")] };
		const result = await executeTaskflow(
			state(def, cwd, "pi-replace-score"),
			deps(cwd, agents, calls, ["append", "replace"]),
		);
		assert.equal(result.ok, true);
		assert.equal(calls.length, 1);
		assert.equal(result.state.phases.quality?.gate?.verdict, "pass");
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("append and default scoring judges remain admissible on append-only hosts", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const agents = [agent("default-judge"), agent("append-judge", "append")];
		const def: Taskflow = {
			name: "append-default-score",
			phases: [
				{ ...scoringGate("append-quality", "append-judge", "default-judge"), final: false },
				{ ...scoringGate("default-quality", undefined, "default-judge"), dependsOn: ["append-quality"] },
			],
		};
		const result = await executeTaskflow(
			state(def, cwd, "append-default-score"),
			deps(cwd, agents, calls, ["append"]),
		);
		assert.equal(result.ok, true);
		assert.equal(calls.length, 2);
		assert.equal(result.state.phases["default-quality"]?.gate?.verdict, "pass");
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("nested statically false with argument excludes a replacement phase from admission", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const child: Taskflow = {
			name: "guarded-replace-child",
			args: { enabled: { type: "boolean", required: true } },
			phases: [{ id: "replace-work", agent: "replace-worker", task: "replace", when: "{args.enabled}", final: true }],
		};
		const parent: Taskflow = {
			name: "guarded-replace-parent",
			phases: [{ id: "nested", type: "flow", use: child.name, with: { enabled: false }, final: true }],
		};
		const result = await executeTaskflow(
			state(parent, cwd, "guarded-replace-false"),
			deps(cwd, [agent("replace-worker", "replace")], calls, ["append"], {
				loadFlow: (name: string) => name === child.name ? child : undefined,
			}),
		);
		assert.equal(result.ok, true);
		assert.equal(calls.length, 0);
		assert.equal(result.state.phases.nested?.status, "done");
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("child defaults participate in nested static admission", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const child: Taskflow = {
			name: "default-disabled-replace-child",
			args: { enabled: { type: "boolean", default: false } },
			phases: [{ id: "replace-work", agent: "replace-worker", task: "replace", when: "{args.enabled}", final: true }],
		};
		const parent: Taskflow = {
			name: "default-disabled-replace-parent",
			phases: [{ id: "nested", type: "flow", use: child.name, final: true }],
		};
		const result = await executeTaskflow(
			state(parent, cwd, "guarded-replace-default-false"),
			deps(cwd, [agent("replace-worker", "replace")], calls, ["append"], {
				loadFlow: (name: string) => name === child.name ? child : undefined,
			}),
		);
		assert.equal(result.ok, true);
		assert.equal(calls.length, 0);
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("nested statically true replacement phase is rejected on append-only hosts", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const child: Taskflow = {
			name: "enabled-replace-child",
			args: { enabled: { type: "boolean", required: true } },
			phases: [{ id: "replace-work", agent: "replace-worker", task: "replace", when: "{args.enabled}", final: true }],
		};
		const parent: Taskflow = {
			name: "enabled-replace-parent",
			phases: [{ id: "nested", type: "flow", use: child.name, with: { enabled: true }, final: true }],
		};
		const result = await executeTaskflow(
			state(parent, cwd, "guarded-replace-true"),
			deps(cwd, [agent("replace-worker", "replace")], calls, ["append"], {
				loadFlow: (name: string) => name === child.name ? child : undefined,
			}),
		);
		assert.equal(result.ok, false);
		assert.equal(calls.length, 0);
		assert.match(result.finalOutput, /systemPromptMode.*replace.*active host/i);
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("nested dynamically unresolved replacement guard remains conservative", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const child: Taskflow = {
			name: "dynamic-replace-child",
			args: { enabled: { type: "boolean", required: true } },
			phases: [{ id: "replace-work", agent: "replace-worker", task: "replace", when: "{args.enabled}", final: true }],
		};
		const parent: Taskflow = {
			name: "dynamic-replace-parent",
			phases: [
				{ id: "decide", agent: "append-worker", task: "decide", output: "json" },
				{
					id: "nested",
					type: "flow",
					use: child.name,
					with: { enabled: "{steps.decide.json.enabled}" },
					dependsOn: ["decide"],
					final: true,
				},
			],
		};
		const result = await executeTaskflow(
			state(parent, cwd, "guarded-replace-dynamic"),
			deps(cwd, [agent("append-worker", "append"), agent("replace-worker", "replace")], calls, ["append"], {
				loadFlow: (name: string) => name === child.name ? child : undefined,
			}),
		);
		assert.equal(result.ok, false);
		assert.equal(calls.length, 0, "conservative root admission must reject before the dynamic producer runs");
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("statically disabled ordinary replacement phase remains ignored", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const def: Taskflow = {
			name: "ordinary-disabled-replace",
			args: { enabled: { type: "boolean", default: false } },
			phases: [{ id: "replace-work", agent: "replace-worker", task: "replace", when: "{args.enabled}", final: true }],
		};
		const result = await executeTaskflow(
			state(def, cwd, "ordinary-disabled-replace"),
			deps(cwd, [agent("replace-worker", "replace")], calls, ["append"]),
		);
		assert.equal(result.ok, true);
		assert.equal(calls.length, 0);
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("separate invocations of one nested flow are admitted with their own static arguments", async () => {
	const cwd = tempDir();
	try {
		const calls: string[] = [];
		const cacheStore = new CacheStore(cwd);
		const child: Taskflow = {
			name: "multi-invocation-replace-child",
			args: { enabled: { type: "boolean", required: true } },
			phases: [{ id: "replace-work", agent: "replace-worker", task: "replace", when: "{args.enabled}", final: true }],
		};
		const parent: Taskflow = {
			name: "multi-invocation-replace-parent",
			phases: [
				{ id: "disabled", type: "flow", use: child.name, with: { enabled: false } },
				{
					id: "enabled",
					type: "flow",
					use: child.name,
					with: { enabled: true },
					dependsOn: ["disabled"],
					cache: { scope: "cross-run" },
					final: true,
				},
			],
		};
		const shared = {
			cacheStore,
			loadFlow: (name: string) => name === child.name ? child : undefined,
		};
		const agents = [agent("replace-worker", "replace")];
		const primed = await executeTaskflow(
			state(parent, cwd, "multi-invocation-prime"),
			deps(cwd, agents, calls, ["append", "replace"], shared),
		);
		assert.equal(primed.ok, true);
		assert.equal(calls.length, 1);

		const rejected = await executeTaskflow(
			state(parent, cwd, "multi-invocation-reject"),
			deps(cwd, agents, calls, ["append"], shared),
		);
		assert.equal(rejected.ok, false);
		assert.equal(calls.length, 1);
		assert.equal(rejected.state.phases.enabled?.status, "failed");
		assert.notEqual(rejected.state.phases.enabled?.cacheHit, "cross-run");
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});
