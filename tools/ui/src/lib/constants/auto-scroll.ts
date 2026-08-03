export const AUTO_SCROLL_INTERVAL = 100;
// Conversation landing: the page keeps growing after the first bottom pin
// without DOM mutations (content-visibility size realizations, syntax
// highlight passes), so the pin repeats every frame until the height holds
// for this many consecutive frames, bounded by the time cap below.
export const LANDING_STABLE_FRAMES = 10;
export const LANDING_SETTLE_MAX_MS = 1000;
// Discrete content injection (e.g. a synthetic cwd-change message appended
// mid-chat): content-visibility sizes the new node at its placeholder before
// the real height is known, so scrollHeight is transiently too large and the
// viewport snaps back once it collapses. Hold the height briefly before the
// pin so the injected row appears without a jump. Smaller than the landing
// settle because a single row settles in a few frames.
export const INJECTION_STABLE_FRAMES = 3;
export const INJECTION_SETTLE_MAX_MS = 400;
// Chat main view: tight threshold because scroll-here events come from
// discrete assistant-message appends.
export const AUTO_SCROLL_AT_BOTTOM_THRESHOLD = 10;
// Reasoning block: stickier because reasoning fires many small
// incremental DOM writes that easily drift a few pixels off bottom.
export const REASONING_SCROLL_AT_BOTTOM_THRESHOLD_PX = 64;
// Syntax-highlighted code: stickier than the chat main view because line
// wrap reflows while the highlight.js pass settles can drift a few pixels
// off bottom.
export const SYNTAX_CODE_SCROLL_AT_BOTTOM_THRESHOLD_PX = 32;
// Streaming tool output (e.g. exec_shell_command): shell commands produce
// lots of small line writes and the exit-code line appended at the tail
// past the last user-visible frame is what triggers DOM drift, so use a
// threshold generous enough to capture that tail flush.
export const TOOL_RUNTIME_SCROLL_AT_BOTTOM_THRESHOLD_PX = 64;
