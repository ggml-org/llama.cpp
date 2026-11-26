/** How far (px) a sheet page slides as it comes in or leaves; nothing with reduced motion. */
export const SHEET_PAGE_SLIDE_PX = 32;

/** How long (ms) the slide between two sheet pages takes. */
export const SHEET_PAGE_SLIDE_MS = 220;

/** Labels of a sheet's corner key. */
export const SHEET_TEXT = {
	BACK: 'Back',
	CLOSE: 'Close'
} as const;

/**
 * Classes of a drawer's content when it holds a SheetHeader: the drag handle sits 0.5rem from the
 * top, clear of the corner key and of the title centered on that key.
 */
export const SHEET_CONTENT_CLASS = '[&>div:first-child]:mt-2';

/** Placeholder of a NativeSelect with no option picked. */
export const NATIVE_SELECT_PLACEHOLDER = 'Select';
