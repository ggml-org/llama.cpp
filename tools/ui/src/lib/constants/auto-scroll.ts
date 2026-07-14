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
