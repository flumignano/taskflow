import type { AgentConfig } from "./agents.ts";
import { interpolateValue } from "./interpolate.ts";
import { classifyWhen } from "./preflight.ts";
import { AGENT_RUNNING_PHASE_TYPES, resolveArgs, type Phase, type Taskflow } from "./schema.ts";
import type { SystemPromptMode } from "./host/runner-types.ts";

/** A host/agent incompatibility found before any executable or reusable result is trusted. */
export interface SystemPromptModeAdmissionFailure {
	/** Root phase whose execution tree contains the incompatible agent. */
	rootPhaseId: string;
	/** Referenced agent-running phase (may belong to a nested flow). */
	phaseId: string;
	agentName: string;
	mode: SystemPromptMode;
	error: string;
}

/** Agent references authored by one phase, including branch and judge overrides. */
function phaseAgentReferences(phase: Phase): Array<string | undefined> {
	const type = phase.type ?? "agent";
	const authored: Array<string | undefined> = [];
	if ((type === "parallel" || type === "race" || type === "tournament") && (phase.branches?.length ?? 0) > 0) {
		for (const branch of phase.branches ?? []) authored.push(branch.agent ?? phase.agent);
	} else {
		authored.push(phase.agent);
	}
	if (type === "tournament") authored.push(phase.judgeAgent ?? phase.agent);
	const scoreJudge =
		type === "gate" && phase.score && typeof phase.score === "object"
			? (phase.score as { judge?: { agent?: string } }).judge
			: undefined;
	if (scoreJudge && typeof scoreJudge === "object") authored.push(scoreJudge.agent ?? phase.agent);
	return authored;
}

/** Match runtime agent resolution: an unknown/omitted name falls back to the first discovered agent. */
function resolveConfiguredAgent(name: string | undefined, agents: readonly AgentConfig[]): AgentConfig | undefined {
	if (name !== undefined) {
		const configured = agents.find((agent) => agent.name === name);
		if (configured) return configured;
	}
	return agents[0];
}

/** Best-effort static normalization for recursively authored inline flow definitions.
 * Runtime-resolved dynamic definitions are checked again at their actual execution boundary. */
function inlineTaskflow(raw: unknown, phaseId: string): Taskflow | undefined {
	let parsed = raw;
	if (typeof parsed === "string") {
		try {
			parsed = JSON.parse(parsed) as unknown;
		} catch {
			return undefined;
		}
	}
	if (Array.isArray(parsed)) {
		return { name: `inline:${phaseId}`, phases: parsed as Phase[] };
	}
	if (!parsed || typeof parsed !== "object") return undefined;
	const candidate = parsed as { name?: unknown; phases?: unknown };
	if (!Array.isArray(candidate.phases)) return undefined;
	return {
		...(parsed as Taskflow),
		name: typeof candidate.name === "string" && candidate.name.length > 0 ? candidate.name : `inline:${phaseId}`,
		phases: candidate.phases as Phase[],
	};
}

/** Resolve only invocation state available before execution. Unresolved step or
 * previous-output placeholders remain intact and therefore classify conservatively. */
function resolveStaticNestedArgs(
	def: Taskflow,
	phase: Phase,
	args: Record<string, unknown>,
): Record<string, unknown> {
	const provided: Record<string, unknown> = {};
	const context = { args, steps: {} };
	for (const [key, value] of Object.entries(phase.with ?? {})) {
		provided[key] = interpolateValue(value, context);
	}
	return resolveArgs(def, provided);
}

/**
 * Validate the system-prompt modes requested by agents referenced from the
 * executable flow closure. Unused discovered profiles are deliberately ignored.
 *
 * Omitted host capability is append-only for backward compatibility. An
 * explicit `replace` request therefore fails closed unless the active host owns
 * and propagates a capability list containing `replace`.
 */
export function systemPromptModeAdmissionFailure(
	def: Pick<Taskflow, "name" | "phases">,
	agents: readonly AgentConfig[],
	systemPromptModes: readonly SystemPromptMode[] | undefined,
	opts: {
		loadFlow?: (name: string) => Taskflow | undefined;
		args?: Record<string, unknown>;
	} = {},
): SystemPromptModeAdmissionFailure | undefined {
	const replaceSupported = systemPromptModes?.includes("replace") === true;
	const activeDefinitions = new Set<object>();
	const activeSavedFlows = new Set<string>();

	const visit = (
		current: Pick<Taskflow, "name" | "phases">,
		path: readonly string[],
		rootPhaseId: string | undefined,
		args: Record<string, unknown>,
	): SystemPromptModeAdmissionFailure | undefined => {
		const currentDefinition = current as object;
		if (activeDefinitions.has(currentDefinition)) return undefined;
		activeDefinitions.add(currentDefinition);
		try {
			for (const phase of current.phases) {
				if (classifyWhen(phase.when, args) === "static-false") continue;
				const type = phase.type ?? "agent";
				const root = rootPhaseId ?? phase.id;
				const phasePath = [...path, phase.id];
				if ((AGENT_RUNNING_PHASE_TYPES as readonly string[]).includes(type)) {
					const resolved = new Map<string, AgentConfig>();
					for (const reference of phaseAgentReferences(phase)) {
						const configured = resolveConfiguredAgent(reference, agents);
						if (configured) resolved.set(configured.name, configured);
					}
					for (const configured of resolved.values()) {
						if (configured.systemPromptMode !== "replace" || replaceSupported) continue;
						const error =
							`Agent '${configured.name}' referenced by phase '${phasePath.join(" -> ")}' requested ` +
							"systemPromptMode 'replace', but the active host does not support replacement system prompts.";
						return {
							rootPhaseId: root,
							phaseId: phase.id,
							agentName: configured.name,
							mode: "replace",
							error,
						};
					}
				}

				if (type !== "flow" && type !== "expand") continue;
				let nested: Taskflow | undefined;
				let activeSavedFlow: string | undefined;
				if (phase.def !== undefined) {
					nested = inlineTaskflow(phase.def, phase.id);
				} else if (type === "flow" && phase.use && opts.loadFlow && !activeSavedFlows.has(phase.use)) {
					activeSavedFlow = phase.use;
					activeSavedFlows.add(activeSavedFlow);
					try {
						nested = opts.loadFlow(activeSavedFlow);
					} catch {
						// Loader failures are handled by the actual flow execution boundary.
						// Capability preflight must not make a statically unused/failed load
						// escape the runtime's fail-soft state closure.
						nested = undefined;
					}
				}
				if (!nested) {
					if (activeSavedFlow) activeSavedFlows.delete(activeSavedFlow);
					continue;
				}
				const nestedArgs = resolveStaticNestedArgs(nested, phase, args);
				try {
					const failure = visit(nested, phasePath, root, nestedArgs);
					if (failure) return failure;
				} finally {
					if (activeSavedFlow) activeSavedFlows.delete(activeSavedFlow);
				}
			}
			return undefined;
		} finally {
			activeDefinitions.delete(currentDefinition);
		}
	};

	return visit(def, [def.name], undefined, opts.args ?? {});
}
