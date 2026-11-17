// only the constants the copied components import, from ui/apps/chat's ui.constants.ts and auto-scroll.constants.ts

/** Data attributes the UI sets on the document. */
export const UI_DATA_ATTRS = {
	HIGHLIGHT_THEME_PREVIEW: 'data-highlight-theme-preview'
} as const;

/** How close to the bottom (px) a streaming code block counts as at the bottom. */
export const SYNTAX_CODE_SCROLL_AT_BOTTOM_THRESHOLD_PX = 32;
