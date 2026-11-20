/**
 * Search dialog constants: how many results each section keeps and how long typing settles
 * before the slower searches (message content, Hugging Face) run.
 */

/** Recent chats shown before typing, so the actions stay in view below them. */
export const CONVERSATION_SEARCH_RECENT_LIMIT = 5;

/** Title matches kept while typing; the rest stays reachable by typing more. */
export const CONVERSATION_SEARCH_MATCH_LIMIT = 50;

/** Chats matched by their messages; the scan stops once this many are found, newest first. */
export const CONVERSATION_SEARCH_CONTENT_LIMIT = 20;

/** Pause after typing before the messages are scanned. */
export const CONVERSATION_SEARCH_CONTENT_DEBOUNCE_MS = 150;

/** Characters kept before the match in a snippet. */
export const CONVERSATION_SEARCH_SNIPPET_LEAD = 24;

/** Snippet length, the match included. */
export const CONVERSATION_SEARCH_SNIPPET_MAX_CHARS = 120;

/** Hugging Face matches kept: Discover lists the rest. */
export const CONVERSATION_SEARCH_HUB_LIMIT = 8;

/** Pause after typing before Hugging Face is asked, as long as the Discover search waits. */
export const CONVERSATION_SEARCH_HUB_DEBOUNCE_MS = 300;
