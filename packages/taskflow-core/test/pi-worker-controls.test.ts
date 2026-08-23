import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { test } from "node:test";
import type { AgentConfig } from "../src/agents.ts";
import { CacheStore } from "../src/cache.ts";
import { phaseFingerprint } from "../src/flowir/index.ts";
import type { RunOptions, RunResult } from "../src/runner-core.ts";
import {
	agentDefinitionsIdentity,
	executeTaskflow,
	type RuntimeDeps,
} from "../src/runtime.ts";
import { validateTaskflow, type Taskflow } from "../src/schema.ts";
import type { RunState } from "../src/store.ts";
import { emptyUsage } from "../src/usage.ts";

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

function ok(agent: string, task: string): RunResult {
	return {
		agent,
		task,
		exitCode: 0,
		output: "done",
		stderr: "",
		usage: emptyUsage(),
		stopReason: "end",
	};
}

function agent(name: string, tools?: string[], systemPromptMode: "append" | "replace" = "append"): AgentConfig {
	return {
		name,
		description: `${name} agent`,
		tools,
		systemPrompt: "Worker prompt.",
		systemPromptMode,
		source: "project",
		filePath: `/agents/${name}.md`,
	};
}

async function executeWithTools(
	agents: AgentConfig[],
	tools: string[] | undefined,
): Promise<{ result: Awaited<ReturnType<typeof executeTaskflow>>; calls: RunOptions[] }> {
	const calls: RunOptions[] = [];
	const def: Taskflow = {
		name: "tool-ceiling",
		phases: [{ id: "work", type: "agent", agent: agents[0].name, task: "work", ...(tools === undefined ? {} : { tools }), final: true }],
	};
	const runTask: RuntimeDeps["runTask"] = async (_cwd, _agents, agentName, task, opts) => {
		calls.push(opts);
		return ok(agentName, task);
	};
	const result = await executeTaskflow(state(def, "/tmp", `tools-${Math.random()}`), {
		cwd: "/tmp",
		agents,
		runTask,
		persist: () => {},
	});
	return { result, calls };
}

test("agent tool ceiling: phase subsets and exact sets are accepted", async () => {
	const declared = agent("reader", ["read", "fffind", "ffgrep"]);
	const subset = await executeWithTools([declared], ["read", "ffgrep"]);
	assert.equal(subset.result.ok, true);
	assert.deepEqual(subset.calls[0]?.tools, ["read", "ffgrep"]);

	const exact = await executeWithTools([declared], ["read", "fffind", "ffgrep"]);
	assert.equal(exact.result.ok, true);
	assert.deepEqual(exact.calls[0]?.tools, ["read", "fffind", "ffgrep"]);
});

test("agent tool ceiling: phase expansion is rejected before the runner is called", async () => {
	const expanded = await executeWithTools([agent("reader", ["read", "fffind", "ffgrep"])], ["read", "bash"]);
	assert.equal(expanded.result.ok, false);
	assert.equal(expanded.calls.length, 0);
	assert.match(expanded.result.finalOutput, /reader.*bash|bash.*reader/i);
	assert.match(expanded.result.finalOutput, /tool.*(ceiling|expansion|unauthorized)/i);
});

test("agent tool ceiling: an agent without declared tools preserves phase.tools semantics", async () => {
	const unrestricted = await executeWithTools([agent("legacy")], ["read", "bash"]);
	assert.equal(unrestricted.result.ok, true);
	assert.deepEqual(unrestricted.calls[0]?.tools, ["read", "bash"]);
});

test("skills validation rejects malformed, duplicate, and non-agent-running selections", () => {
	for (const skills of [["../escape"], ["Skill-A"], ["skill-a", "skill-a"]]) {
		const result = validateTaskflow({
			name: "invalid-skills",
			phases: [{ id: "work", type: "agent", task: "work", skills, final: true }],
		});
		assert.equal(result.ok, false, JSON.stringify(skills));
		assert.match(result.errors.join("\n"), /skills/i);
	}
	const script = validateTaskflow({
		name: "script-skills",
		phases: [{ id: "work", type: "script", run: "echo ok", skills: [], final: true }],
	});
	assert.equal(script.ok, false);
	assert.match(script.errors.join("\n"), /skills.*agent-running/i);
});

test("explicit skills fail closed on a host without a skill resolver", async () => {
	let calls = 0;
	const def: Taskflow = {
		name: "unsupported-skills",
		phases: [{ id: "work", type: "agent", agent: "reader", task: "work", skills: [], final: true }],
	};
	const result = await executeTaskflow(state(def, "/tmp", "unsupported-skills"), {
		cwd: "/tmp",
		agents: [agent("reader", ["read"])],
		runTask: async (_cwd, _agents, agentName, task) => {
			calls++;
			return ok(agentName, task);
		},
	});
	assert.equal(result.ok, false);
	assert.equal(calls, 0);
	assert.match(result.state.phases.work.error ?? "", /skills.*not supported|host.*skills/i);
});

test("resolved skill content and resolution participate in cross-run cache identity", async () => {
	const cwd = fs.mkdtempSync(path.join(os.tmpdir(), "tf-skill-cache-"));
	try {
		const cacheStore = new CacheStore(cwd);
		const def: Taskflow = {
			name: "skill-cache",
			phases: [{
				id: "work",
				type: "agent",
				agent: "reader",
				task: "work",
				skills: ["skill-a"],
				cache: { scope: "cross-run" },
				final: true,
			}],
		};
		let contentHash = "body-v1";
		let calls = 0;
		const deps: RuntimeDeps = {
			cwd,
			agents: [agent("reader", ["read"])],
			cacheStore,
			resolveSkills: (names) => names.map((name) => ({
				name,
				filePath: path.join(cwd, ".pi", "skills", name, "SKILL.md"),
				contentHash,
			})),
			runTask: async (_cwd, _agents, agentName, task, opts) => {
				calls++;
				assert.equal(opts.skills?.[0]?.contentHash, contentHash);
				return ok(agentName, task);
			},
		};

		await executeTaskflow(state(def, cwd, "skill-cache-1"), deps);
		const same = await executeTaskflow(state(def, cwd, "skill-cache-2"), deps);
		assert.equal(calls, 1);
		assert.equal(same.state.phases.work.cacheHit, "cross-run");

		contentHash = "body-v2";
		const changed = await executeTaskflow(state(def, cwd, "skill-cache-3"), deps);
		assert.equal(changed.ok, true);
		assert.equal(calls, 2, "changed skill body must miss the old cache entry");

		contentHash = "body-v2";
		deps.resolveSkills = (names) => names.map((name) => ({
			name,
			filePath: path.join(cwd, ".pi", "skills-alt", name, "SKILL.md"),
			contentHash,
		}));
		await executeTaskflow(state(def, cwd, "skill-cache-4"), deps);
		assert.equal(calls, 3, "changed exact resolution must miss the old cache entry");
	} finally {
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("prompt mode and selected skill names affect structural identity", async () => {
	const appendIdentity = agentDefinitionsIdentity([agent("reader", ["read"], "append")]);
	const replaceIdentity = agentDefinitionsIdentity([agent("reader", ["read"], "replace")]);
	assert.notEqual(appendIdentity, replaceIdentity);
	const legacy = agent("reader", ["read"], "append");
	delete legacy.systemPromptMode;
	assert.equal(appendIdentity, agentDefinitionsIdentity([legacy]), "default/explicit append keep the legacy identity");

	const phase = (skills?: string[]): Taskflow => ({
		name: "skill-fingerprint",
		phases: [{ id: "work", type: "agent", agent: "reader", task: "work", ...(skills === undefined ? {} : { skills }), final: true }],
	});
	assert.notEqual(await phaseFingerprint(phase(), "work"), await phaseFingerprint(phase([]), "work"));
	assert.notEqual(await phaseFingerprint(phase(["skill-a"]), "work"), await phaseFingerprint(phase(["skill-b"]), "work"));
});
