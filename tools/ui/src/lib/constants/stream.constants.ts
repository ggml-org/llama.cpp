// grace window after a visibilitychange before we kick a reader whose socket likely died
// while the tab was hidden. covers brief background pauses without thrashing live streams
export const STREAM_VISIBILITY_KICK_MS = 3000;

// separator joining a conversation id and its per-model stream identity
// suffix (conv::model) used by the server side replay buffer
export const CONVERSATION_ID_SEPARATOR = '::';

/**
 * Server-sent events wire format, shared by the chat stream and the
 * /models/sse status feed (text/event-stream).
 */

// blank line between two events
export const SSE_RECORD_SEPARATOR = '\n\n';

// line break inside an event
export const SSE_LINE_SEPARATOR = '\n';

// data field prefix, the value follows after an optional space
export const SSE_DATA_PREFIX = 'data:';

// end-of-stream marker on the chat completion stream
export const SSE_DONE_MARKER = '[DONE]';
