export const INITIAL_FILE_SIZE = 0;
export const PROMPT_CONTENT_SEPARATOR = '\n\n';
export const CLIPBOARD_CONTENT_QUOTE_PREFIX = '"';
export const PROMPT_TRIGGER_PREFIX = '/';
export const NEW_CHAT_DRAFT_KEY = '__new_chat__';
/** Lines of text from which the main chat form expands into its large editor. */
export const CHAT_FORM_EXPAND_MIN_LINES = 4;
/** Below this text height (a 24px line and change) the main chat form keeps to one row. */
export const CHAT_FORM_SINGLE_ROW_MAX_TEXT_PX = 36;

/**
 * How far the chat form floats above the bottom of the screen: a phone browser's toolbar is
 * already a margin below it (iOS Safari's most of all), and the installed app keeps it clear
 * of the home indicator, in line with the bottom corner action.
 */
export const CHAT_FORM_BOTTOM_OFFSET = {
	desktop: '1rem',
	installedApp: '1.5rem',
	iosSafari: '0.5rem',
	phoneBrowser: '0.75rem'
} as const;

/**
 * The chat form moving between its centered (new chat) and bottom positions:
 * it fades out drifting toward where it goes, then fades back in where it
 * lands, instead of sliding across the column. The exit uses a gentler curve
 * so the form stays visible while it travels; the entry settles quickly.
 */
export const CHAT_FORM_MOVE_OUT_MS = 100;
export const CHAT_FORM_MOVE_IN_MS = 180;
// opening a chat moves the form to the bottom: quicker, so the chat is
// there right away
export const CHAT_FORM_MOVE_TO_BOTTOM_OUT_MS = 90;
export const CHAT_FORM_MOVE_TO_BOTTOM_IN_MS = 160;
export const CHAT_FORM_MOVE_SHIFT_PX = 10;
/** Share of the exit by which the form is gone: it vanishes just before the end of its drift. */
export const CHAT_FORM_MOVE_OUT_FADE_END = 0.8;
export const CHAT_FORM_MOVE_OUT_EASING = 'cubic-bezier(0.33, 1, 0.68, 1)';
export const CHAT_FORM_MOVE_IN_EASING = 'cubic-bezier(0.23, 1, 0.32, 1)';

/**
 * The new-chat greeting fades in as the form settles: it starts
 * halfway through the form's fade-in so both read as one motion.
 */
export const CHAT_GREETING_ENTER_DELAY_MS = CHAT_FORM_MOVE_IN_MS / 2;

/**
 * The main chat form growing into its expanded editor and back (the
 * `duration-400` of its height and position transitions). After a collapse
 * the greeting waits for it, so it fades in over a form that has settled.
 */
export const CHAT_FORM_RESIZE_MS = 400;

/**
 * On a phone the expanded form is a full-screen sheet, which moves more slowly
 * (see useChatFormExpansion).
 */
export const CHAT_FORM_SHEET_MS = 500;

/**
 * An ease-out without overshoot, for moves that must land exactly: the phone
 * sheet, and the new chat's form, whose top edge an overshoot would lift.
 */
export const CHAT_FORM_EASE_OUT = 'cubic-bezier(0.32, 0.72, 0, 1)';

/** The phone's corner keys and header stepping aside for the expanded form, and back. */
export const CHAT_FORM_CHROME_FADE_MS = 150;
