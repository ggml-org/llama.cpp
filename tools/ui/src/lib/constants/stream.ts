// grace window after a visibilitychange before we kick a reader whose socket likely died
// while the tab was hidden. covers brief background pauses without thrashing live streams
export const STREAM_VISIBILITY_KICK_MS = 3000;
