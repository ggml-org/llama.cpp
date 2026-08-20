/* Conversation compaction constants */
import { CONVERSATION_ID_SEPARATOR } from './stream.constants';

const SUMMARY_PLACEHOLDER = '{{SUMMARY}}';

export const COMPACTION = {
	CONTEXT_TEMPLATE: `This conversation was compacted. Summary of the earlier exchange:\n\n${SUMMARY_PLACEHOLDER}`,
	DEFAULT_PROMPT:
		'Summarize this conversation so it can seamlessly continue in a fresh context. Capture: the overall goal, key facts and decisions, the current state of any ongoing work (including code, file names, and data discussed), and what remains to be done. Write the summary as dense prose addressed to the assistant that will resume the conversation. Return ONLY the summary, nothing else.',
	// Server stream sessions are matched by exact id or by the id::* prefix,
	// so a prefixed identity keeps the summary stream inside the visibility
	// kick and byte-offset resume machinery while the reload probe for the
	// bare conversation id can never reattach it as a chat turn.
	STREAM_ID_PREFIX: `compact${CONVERSATION_ID_SEPARATOR}`,
	SUMMARY_PLACEHOLDER
} as const;
