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
