/**
 * Settings key constants for ChatSettings configuration.
 *
 * These keys correspond to properties in SettingsConfigType and are used
 * in settings field configurations to ensure consistency.
 */
export const SETTINGS_KEYS = {
	// General
	THEME: 'theme',
	API_KEY: 'apiKey',
	SYSTEM_MESSAGE: 'systemMessage',
	PASTE_LONG_TEXT_TO_FILE_LEN: 'pasteLongTextToFileLen',
	COPY_TEXT_ATTACHMENTS_AS_PLAIN_TEXT: 'copyTextAttachmentsAsPlainText',
	ENABLE_CONTINUE_GENERATION: 'enableContinueGeneration',
	PDF_AS_IMAGE: 'pdfAsImage',
	ASK_FOR_TITLE_CONFIRMATION: 'askForTitleConfirmation',
	// Display
	SHOW_MESSAGE_STATS: 'showMessageStats',
	SHOW_THOUGHT_IN_PROGRESS: 'showThoughtInProgress',
	KEEP_STATS_VISIBLE: 'keepStatsVisible',
	AUTO_MIC_ON_EMPTY: 'autoMicOnEmpty',
	RENDER_USER_CONTENT_AS_MARKDOWN: 'renderUserContentAsMarkdown',
	DISABLE_AUTO_SCROLL: 'disableAutoScroll',
	ALWAYS_SHOW_SIDEBAR_ON_DESKTOP: 'alwaysShowSidebarOnDesktop',
	AUTO_SHOW_SIDEBAR_ON_NEW_CHAT: 'autoShowSidebarOnNewChat',
	FULL_HEIGHT_CODE_BLOCKS: 'fullHeightCodeBlocks',
	// Sampling
	TEMPERATURE: 'temperature',
	DYNATEMP_RANGE: 'dynatemp_range',
	DYNATEMP_EXPONENT: 'dynatemp_exponent',
	TOP_K: 'top_k',
	TOP_P: 'top_p',
	MIN_P: 'min_p',
	XTC_PROBABILITY: 'xtc_probability',
	XTC_THRESHOLD: 'xtc_threshold',
	TYP_P: 'typ_p',
	MAX_TOKENS: 'max_tokens',
	SAMPLERS: 'samplers',
	BACKEND_SAMPLING: 'backend_sampling',
	// Penalties
	REPEAT_LAST_N: 'repeat_last_n',
	REPEAT_PENALTY: 'repeat_penalty',
	PRESENCE_PENALTY: 'presence_penalty',
	FREQUENCY_PENALTY: 'frequency_penalty',
	DRY_MULTIPLIER: 'dry_multiplier',
	DRY_BASE: 'dry_base',
	DRY_ALLOWED_LENGTH: 'dry_allowed_length',
	DRY_PENALTY_LAST_N: 'dry_penalty_last_n',
	// Developer
	DISABLE_REASONING_PARSING: 'disableReasoningParsing',
	SHOW_RAW_OUTPUT_SWITCH: 'showRawOutputSwitch',
	CUSTOM: 'custom'
} as const;
