import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";
import { getAgentDir, loadSkills } from "@earendil-works/pi-coding-agent";
import {
	findProjectDotPiDir,
	parseFrontmatter,
	type ResolvedSkill,
} from "taskflow-core";

const SKILL_NAME_RE = /^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$/;

function isWithin(root: string, candidate: string): boolean {
	const relative = path.relative(root, candidate);
	return relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative));
}

function canonicalFile(filePath: string): string {
	return fs.realpathSync.native(filePath);
}

/** Discover exactly the project-owned Pi skill files below one trusted `.pi/skills`. */
function projectSkillFiles(cwd: string): string[] {
	const project = findProjectDotPiDir(cwd);
	if (!project) throw new Error(`Explicit Pi skills require a project-owned .pi directory reachable from '${cwd}'`);
	const skillsDir = path.join(project.dotPiDir, "skills");
	let root: string;
	try {
		root = canonicalFile(skillsDir);
		if (!fs.statSync(root).isDirectory()) throw new Error("not a directory");
	} catch (error) {
		throw new Error(`Explicit Pi skills require an accessible project skill directory '${skillsDir}': ${error instanceof Error ? error.message : String(error)}`);
	}
	const projectRoot = canonicalFile(project.projectDir);
	if (!isWithin(projectRoot, root)) {
		throw new Error(`Project skill directory escapes its project boundary: '${skillsDir}' -> '${root}'`);
	}

	const candidates: string[] = [];
	const seenDirectories = new Set<string>();
	const scan = (directory: string, includeRootMarkdown: boolean): void => {
		const canonicalDirectory = canonicalFile(directory);
		if (!isWithin(root, canonicalDirectory)) {
			throw new Error(`Project skill directory escapes '${root}': '${directory}' -> '${canonicalDirectory}'`);
		}
		if (seenDirectories.has(canonicalDirectory)) return;
		seenDirectories.add(canonicalDirectory);

		let entries: fs.Dirent[];
		try {
			entries = fs.readdirSync(canonicalDirectory, { withFileTypes: true })
				.sort((a, b) => a.name.localeCompare(b.name));
		} catch (error) {
			throw new Error(`Cannot read project skill directory '${canonicalDirectory}': ${error instanceof Error ? error.message : String(error)}`);
		}
		for (const entry of entries) {
			if (entry.isSymbolicLink()) {
				throw new Error(`Symbolic links are not allowed in explicit project skill discovery: '${path.join(canonicalDirectory, entry.name)}'`);
			}
		}

		const skillEntry = entries.find((entry) => entry.name === "SKILL.md" && entry.isFile());
		if (skillEntry) {
			candidates.push(canonicalFile(path.join(canonicalDirectory, skillEntry.name)));
			return;
		}

		for (const entry of entries) {
			if (entry.name.startsWith(".") || entry.name === "node_modules") continue;
			const entryPath = path.join(canonicalDirectory, entry.name);
			if (entry.isDirectory()) {
				scan(entryPath, false);
			} else if (includeRootMarkdown && entry.isFile() && entry.name.endsWith(".md")) {
				candidates.push(canonicalFile(entryPath));
			}
		}
	};
	scan(root, true);
	return candidates;
}

function loadedProjectSkills(cwd: string): Map<string, Array<{ filePath: string; contentHash: string }>> {
	const byName = new Map<string, Array<{ filePath: string; contentHash: string }>>();
	for (const candidate of projectSkillFiles(cwd)) {
		const result = loadSkills({
			cwd,
			agentDir: getAgentDir(),
			skillPaths: [candidate],
			includeDefaults: false,
		});
		if (result.skills.length !== 1) continue;
		const skill = result.skills[0];
		const filePath = canonicalFile(skill.filePath);
		const content = fs.readFileSync(filePath, "utf8");
		const declaredName = parseFrontmatter(content).frontmatter.name;
		if (typeof declaredName !== "string" || declaredName !== skill.name) continue;
		const entry = { filePath, contentHash: createHash("sha256").update(content, "utf8").digest("hex") };
		const matches = byName.get(skill.name) ?? [];
		matches.push(entry);
		byName.set(skill.name, matches);
	}
	return byName;
}

/** Resolve authored Pi skill names to exact canonical project files, fail-closed. */
export function resolveProjectPiSkills(names: readonly string[], cwd: string): ResolvedSkill[] {
	if (names.length === 0) return [];
	const seen = new Set<string>();
	for (const name of names) {
		if (!SKILL_NAME_RE.test(name) || name.includes("--")) {
			throw new Error(`Invalid Pi skill name '${name}' (expected 1-64 lowercase letters, numbers, or single hyphens)`);
		}
		if (seen.has(name)) throw new Error(`Duplicate Pi skill selection '${name}'`);
		seen.add(name);
	}

	const available = loadedProjectSkills(cwd);
	return names.map((name) => {
		const matches = available.get(name) ?? [];
		if (matches.length === 0) {
			throw new Error(`Unknown explicit Pi skill '${name}' in the project skill directory for '${cwd}'`);
		}
		if (matches.length > 1) {
			throw new Error(`Ambiguous explicit Pi skill '${name}': ${matches.map((match) => match.filePath).join(", ")}`);
		}
		return { name, ...matches[0] };
	});
}
