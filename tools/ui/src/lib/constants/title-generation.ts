/* Title generation constants */
export const TITLE_GENERATION = {
	MIN_LENGTH: 3,
	FALLBACK: 'New Chat',
	DEFAULT_PROMPT:
		'Based on the following interaction, generate a short, concise title (maximum 6-8 words) that captures the main topic. Return ONLY the title text, nothing else. Do not use quotes.\n\nUser: {{USER}}\n\nAssistant: {{ASSISTANT}}\n\nTitle:',
	PREFIX_PATTERN: /^(Title:|Subject:|Topic:)\s*/i,
	QUOTE_PATTERN: /^["]|["]$/g
} as const;
