// grace window after a visibilitychange before we kick a reader whose socket likely died
// while the tab was hidden. covers brief background pauses without thrashing live streams
export const STREAM_VISIBILITY_KICK_MS = 1000;

// marks the end of an SSE completion stream, the server sends it as the final data payload
export const SSE_DONE_MARKER = '[DONE]';

// prefix of an SSE data line, the payload starts right after it
export const SSE_DATA_PREFIX = 'data: ';
