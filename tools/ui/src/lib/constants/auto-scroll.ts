export const AUTO_SCROLL_INTERVAL = 100;
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
