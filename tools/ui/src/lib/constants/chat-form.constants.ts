/** Data attribute that tags ChatFormInputRich code spans and blocks. */
export const CODE_TOKEN_ATTR = 'data-code-token';

export const INITIAL_FILE_SIZE = 0;
export const PROMPT_CONTENT_SEPARATOR = '\n\n';
export const CLIPBOARD_CONTENT_QUOTE_PREFIX = '"';
export const PROMPT_TRIGGER_PREFIX = '/';
export const NEW_CHAT_DRAFT_KEY = '__new_chat__';

/** Sent prompts kept for ArrowUp / swipe recall. */
export const PROMPT_HISTORY_MAX_ENTRIES = 100;
/** Minimum vertical pointer travel (px) to treat a gesture as history swipe. */
export const PROMPT_HISTORY_SWIPE_MIN_PX = 48;
/** Minimum ms between mouse-wheel history steps. */
export const PROMPT_HISTORY_WHEEL_COOLDOWN_MS = 90;
