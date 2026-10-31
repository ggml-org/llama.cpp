export const INITIAL_FILE_SIZE = 0;
export const PROMPT_CONTENT_SEPARATOR = '\n\n';
export const CLIPBOARD_CONTENT_QUOTE_PREFIX = '"';
export const PROMPT_TRIGGER_PREFIX = '/';
export const NEW_CHAT_DRAFT_KEY = '__new_chat__';
/** Lines of text from which the main chat form expands into its large editor. */
export const CHAT_FORM_EXPAND_MIN_LINES = 4;

/**
 * The chat form moving between its centered (new chat) and bottom positions:
 * it fades out drifting toward where it goes, then fades back in where it
 * lands, instead of sliding across the column. The exit uses a gentler curve
 * so the form stays visible while it travels; the entry settles quickly.
 */
export const CHAT_FORM_MOVE_OUT_MS = 140;
export const CHAT_FORM_MOVE_IN_MS = 180;
// opening a chat moves the form to the bottom: quicker, so the chat is
// there right away
export const CHAT_FORM_MOVE_TO_BOTTOM_OUT_MS = 80;
export const CHAT_FORM_MOVE_TO_BOTTOM_IN_MS = 120;
export const CHAT_FORM_MOVE_SHIFT_PX = 16;
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
 * On a phone the expanded form is a full-screen sheet that moves by transform,
 * more slowly and without the overshoot (see ChatForm's animateSheet).
 */
export const CHAT_FORM_SHEET_MS = 500;
