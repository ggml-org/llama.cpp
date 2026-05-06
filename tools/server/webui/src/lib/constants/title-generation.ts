export const TITLE_MAX_LENGTH = 60;
export const TITLE_TRUNCATE_AT = 57;
export const TITLE_TRUNCATION_SUFFIX = '...';
export const TITLE_MIN_LENGTH = 3;

/** Default fallback title when no meaningful content is available */
export const TITLE_FALLBACK = 'New Chat';

/** Default prompt template for LLM based title generation */
export const TITLE_DEFAULT_PROMPT =
	'Based on the following interaction, generate a short, concise title (maximum 6-8 words) that captures the main topic. Return ONLY the title text, nothing else. Do not use quotes.\n\nUser: {{USER}}\n\nAssistant: {{ASSISTANT}}\n\nTitle:';

/** Patterns to strip leading title prefixes (e.g., "Title:", "Subject:", "Topic:") and quotes */
export const TITLE_PREFIX_PATTERN = /^(Title:|Subject:|Topic:)\s*/i;
export const TITLE_QUOTE_PATTERN = /^["]|["]$/g;
