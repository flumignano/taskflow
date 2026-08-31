import { emptyUsage, type AgentConfig, type RunResult } from "taskflow-core";

/** Reject an explicit replacement prompt request on a host with append-only semantics. */
export function unsupportedSystemPromptModeResult(
	agent: AgentConfig,
	task: string,
	host: string,
): RunResult | undefined {
	if (agent.systemPromptMode !== "replace") return undefined;
	const message = `Agent '${agent.name}' requested systemPromptMode 'replace', but active host '${host}' does not support replacement system prompts.`;
	return {
		agent: agent.name,
		task,
		exitCode: 1,
		output: "",
		stderr: message,
		usage: emptyUsage(),
		stopReason: "error",
		errorMessage: message,
	};
}
