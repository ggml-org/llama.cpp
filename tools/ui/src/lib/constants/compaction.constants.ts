/* Conversation compaction constants */
const SUMMARY_PLACEHOLDER = '{{SUMMARY}}';

export const COMPACTION = {
	CONTEXT_TEMPLATE: `This conversation was compacted. Summary of the earlier exchange:\n\n${SUMMARY_PLACEHOLDER}`,
	DEFAULT_PROMPT:
		'Summarize this conversation so it can seamlessly continue in a fresh context. Capture: the overall goal, key facts and decisions, the current state of any ongoing work (including code, file names, and data discussed), and what remains to be done. Write the summary as dense prose addressed to the assistant that will resume the conversation. Return ONLY the summary, nothing else.',
	SUMMARY_PLACEHOLDER
} as const;
