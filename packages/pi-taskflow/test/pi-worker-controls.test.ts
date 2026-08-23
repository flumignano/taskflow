import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { test } from "node:test";
import type { AgentConfig, ResolvedSkill } from "taskflow-core";
import { CTX_TOOL_NAMES, runAgentTask } from "../src/runner.ts";
import { resolveProjectPiSkills } from "../src/skills.ts";

function writeSkill(root: string, directory: string, name: string, body: string): string {
	const dir = path.join(root, ".pi", "skills", directory);
	fs.mkdirSync(dir, { recursive: true });
	const filePath = path.join(dir, "SKILL.md");
	fs.writeFileSync(filePath, `---\nname: ${name}\ndescription: Smoke skill ${name}.\n---\n\n${body}\n`, "utf8");
	return filePath;
}

function resolved(name: string, filePath: string, contentHash = `hash-${name}`): ResolvedSkill {
	return { name, filePath, contentHash };
}

function worker(systemPromptMode?: "append" | "replace", tools?: string[]): AgentConfig {
	return {
		name: "worker",
		description: "minimal worker",
		tools,
		systemPrompt: "MINIMAL-WORKER-PROMPT",
		...(systemPromptMode === undefined ? {} : { systemPromptMode }),
		source: "project",
		filePath: "/agents/worker.md",
	};
}

async function invokeAndCapture(
	cwd: string,
	capture: string,
	agent: AgentConfig,
	opts: Parameters<typeof runAgentTask>[4],
): Promise<{ argv: string[]; appendPrompt: string | null; systemPrompt: string | null }> {
	fs.rmSync(capture, { force: true });
	const result = await runAgentTask(cwd, [agent], agent.name, "run smoke", opts);
	assert.equal(result.exitCode, 0, result.errorMessage ?? result.stderr);
	return JSON.parse(fs.readFileSync(capture, "utf8")) as {
		argv: string[];
		appendPrompt: string | null;
		systemPrompt: string | null;
	};
}

test("Pi launch controls: prompt mode and explicit skills build exact argv", async () => {
	const cwd = fs.mkdtempSync(path.join(os.tmpdir(), "tf-pi-controls-"));
	const capture = path.join(cwd, "capture.json");
	const fakePi = path.join(cwd, "fake-pi.mjs");
	fs.writeFileSync(
		fakePi,
		`#!${process.execPath}\n` +
		`import * as fs from "node:fs";\n` +
			`const argv = process.argv.slice(2);\n` +
			`const readPrompt = (flag) => { const i = argv.indexOf(flag); return i < 0 ? null : fs.readFileSync(argv[i + 1], "utf8"); };\n` +
			`fs.writeFileSync(${JSON.stringify(capture)}, JSON.stringify({ argv, appendPrompt: readPrompt("--append-system-prompt"), systemPrompt: readPrompt("--system-prompt") }));\n` +
			`const emit = (value) => process.stdout.write(JSON.stringify(value) + "\\n");\n` +
			`emit({type:"agent_start"}); emit({type:"turn_start"});\n` +
			`emit({type:"message_end",message:{role:"assistant",content:[{type:"text",text:"DONE"}],stopReason:"stop"}});\n` +
			`emit({type:"agent_end",willRetry:false}); emit({type:"agent_settled"});\n`,
		{ mode: 0o755 },
	);
	fs.chmodSync(fakePi, 0o755);
	const previous = process.env.PI_TASKFLOW_PI_BIN;
	process.env.PI_TASKFLOW_PI_BIN = fakePi;
	try {
		const legacy = await invokeAndCapture(cwd, capture, worker(), {});
		assert.equal(legacy.appendPrompt, "MINIMAL-WORKER-PROMPT");
		assert.equal(legacy.systemPrompt, null);
		assert.equal(legacy.argv.includes("--no-skills"), false, "omitted skills preserve ambient discovery");

		const append = await invokeAndCapture(cwd, capture, worker("append"), {});
		assert.equal(append.appendPrompt, "MINIMAL-WORKER-PROMPT");
		assert.equal(append.systemPrompt, null);

		const replace = await invokeAndCapture(cwd, capture, worker("replace"), {});
		assert.equal(replace.systemPrompt, "MINIMAL-WORKER-PROMPT");
		assert.equal(replace.appendPrompt, null, "replacement must not duplicate the agent body through append mode");
		assert.equal(replace.argv.includes("--no-context-files"), false);
		assert.equal(replace.argv.includes("--no-skills"), false);

		const none = await invokeAndCapture(cwd, capture, worker(), { skills: [] });
		assert.equal(none.argv.includes("--no-skills"), true);
		assert.equal(none.argv.includes("--skill"), false);

		const noTools = await invokeAndCapture(cwd, capture, worker(undefined, ["read"]), { tools: [] });
		assert.equal(noTools.argv.includes("--no-tools"), true, "empty subset must not broaden to Pi defaults");
		assert.equal(noTools.argv.includes("--tools"), false);

		const firstPath = path.join(cwd, "skill-a", "SKILL.md");
		const one = await invokeAndCapture(cwd, capture, worker(), {
			skills: [resolved("skill-a", firstPath)],
		});
		assert.equal(one.argv.includes("--no-skills"), true);
		const oneSkillIndex = one.argv.indexOf("--skill");
		assert.equal(one.argv[oneSkillIndex + 1], firstPath);

		const smoke = await invokeAndCapture(cwd, capture, worker("replace", ["read"]), {
			tools: ["read"],
			skills: [resolved("skill-a", firstPath)],
		});
		assert.equal(smoke.systemPrompt, "MINIMAL-WORKER-PROMPT");
		assert.equal(smoke.appendPrompt, null);
		assert.equal(smoke.argv[smoke.argv.indexOf("--tools") + 1], "read");
		assert.equal(smoke.argv.includes("--no-skills"), true);
		assert.equal(smoke.argv[smoke.argv.indexOf("--skill") + 1], firstPath);

		const secondPath = path.join(cwd, "skill-b", "SKILL.md");
		const many = await invokeAndCapture(cwd, capture, worker(), {
			skills: [resolved("skill-a", firstPath), resolved("skill-b", secondPath)],
		});
		const selected = many.argv.flatMap((arg, index) => arg === "--skill" ? [many.argv[index + 1]] : []);
		assert.deepEqual(selected, [firstPath, secondPath]);
		assert.equal(many.argv.includes("--no-skills"), true);

		const ctxDir = path.join(cwd, "shared-context");
		const sharingOpts = { tools: ["read"], ctxDir, nodeId: "phase:work" };

		fs.rmSync(capture, { force: true });
		const contextExpansion = await runAgentTask(
			cwd,
			[worker(undefined, ["read"])],
			"worker",
			"share context",
			sharingOpts,
		);
		assert.equal(contextExpansion.exitCode, 1);
		assert.match(contextExpansion.errorMessage ?? contextExpansion.stderr, /context-sharing/i);
		for (const tool of CTX_TOOL_NAMES) {
			assert.match(contextExpansion.errorMessage ?? contextExpansion.stderr, new RegExp(`\\b${tool}\\b`));
		}
		assert.equal(fs.existsSync(capture), false, "unauthorized context tools must fail before child spawn");

		const completeContextEnvelope = ["read", ...CTX_TOOL_NAMES];
		const contextAllowed = await invokeAndCapture(
			cwd,
			capture,
			worker(undefined, completeContextEnvelope),
			sharingOpts,
		);
		assert.equal(
			contextAllowed.argv[contextAllowed.argv.indexOf("--tools") + 1],
			completeContextEnvelope.join(","),
		);

		const legacyContext = await invokeAndCapture(cwd, capture, worker(), {
			ctxDir,
			nodeId: "phase:legacy",
		});
		assert.equal(legacyContext.argv.includes("--tools"), false);
		assert.equal(legacyContext.argv.includes("--no-tools"), false);
		assert.equal(legacyContext.argv.includes("--extension"), true);

		fs.rmSync(capture, { force: true });
		const expanded = await runAgentTask(cwd, [worker(undefined, ["read"])], "worker", "expand", {
			tools: ["read", "bash"],
		});
		assert.equal(expanded.exitCode, 1);
		assert.match(expanded.errorMessage ?? expanded.stderr, /(ceiling|unauthorized|expansion).*bash|bash.*(ceiling|unauthorized|expansion)/i);
		assert.equal(fs.existsSync(capture), false, "unauthorized expansion must fail before child spawn");
	} finally {
		if (previous === undefined) delete process.env.PI_TASKFLOW_PI_BIN;
		else process.env.PI_TASKFLOW_PI_BIN = previous;
		fs.rmSync(cwd, { recursive: true, force: true });
	}
});

test("Pi skill resolver: exact names, deterministic order, and content identity", () => {
	const root = fs.mkdtempSync(path.join(os.tmpdir(), "tf-skill-resolve-"));
	try {
		const child = path.join(root, "packages", "app");
		fs.mkdirSync(child, { recursive: true });
		const aPath = writeSkill(root, "directory-a", "skill-a", "Body A v1.");
		const bPath = writeSkill(root, "directory-b", "skill-b", "Body B.");

		const resolvedSkills = resolveProjectPiSkills(["skill-b", "skill-a"], child);
		assert.deepEqual(resolvedSkills.map((skill) => skill.name), ["skill-b", "skill-a"]);
		assert.deepEqual(resolvedSkills.map((skill) => skill.filePath), [fs.realpathSync(bPath), fs.realpathSync(aPath)]);
		assert.match(resolvedSkills[0].contentHash, /^[0-9a-f]{64}$/);

		const before = resolveProjectPiSkills(["skill-a"], child)[0].contentHash;
		fs.writeFileSync(aPath, `---\nname: skill-a\ndescription: Smoke skill skill-a.\n---\n\nBody A v2.\n`, "utf8");
		const after = resolveProjectPiSkills(["skill-a"], child)[0].contentHash;
		assert.notEqual(before, after, "SKILL.md body changes must change content identity");
	} finally {
		fs.rmSync(root, { recursive: true, force: true });
	}
});

test("Pi skill resolver: unknown, ambiguous, malformed, and escaping selections fail closed", () => {
	const root = fs.mkdtempSync(path.join(os.tmpdir(), "tf-skill-reject-"));
	try {
		writeSkill(root, "one", "same-name", "One.");
		writeSkill(root, "two", "same-name", "Two.");
		assert.throws(() => resolveProjectPiSkills(["missing"], root), /unknown.*missing/i);
		assert.throws(() => resolveProjectPiSkills(["same-name"], root), /ambiguous.*same-name/i);
		for (const invalid of ["../escape", "/absolute", "Uppercase", "double--dash"]) {
			assert.throws(() => resolveProjectPiSkills([invalid], root), /invalid.*skill/i);
		}

		const outside = fs.mkdtempSync(path.join(os.tmpdir(), "tf-skill-outside-"));
		try {
			writeSkill(outside, "outside", "outside", "Outside.");
			const skillsRoot = path.join(root, ".pi", "skills");
			fs.symlinkSync(path.join(outside, ".pi", "skills", "outside"), path.join(skillsRoot, "escape"), "dir");
			assert.throws(() => resolveProjectPiSkills(["outside"], root), /symbolic link|escape/i);
		} finally {
			fs.rmSync(outside, { recursive: true, force: true });
		}
	} finally {
		fs.rmSync(root, { recursive: true, force: true });
	}
});
