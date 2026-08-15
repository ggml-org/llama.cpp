/**
 * Settings key constants for ChatSettings configuration.
 *
 * These keys correspond to properties in SettingsConfigType and are used
 * in settings field configurations to ensure consistency.
 */
export const SETTINGS_KEYS = {
	AGENTIC_MAX_TURNS: 'agenticMaxTurns',
	ALWAYS_SHOW_SIDEBAR_ON_DESKTOP: 'alwaysShowSidebarOnDesktop',
	ALWAYS_SHOW_TOOL_CALL_CONTENT: 'alwaysShowToolCallContent',
	API_KEY: 'apiKey',
	AUTO_MIC_ON_EMPTY: 'autoMicOnEmpty',
	BACKEND_SAMPLING: 'backend_sampling',
	CONVERSATION_TABS: 'conversationTabs',
	COPY_TEXT_ATTACHMENTS_AS_PLAIN_TEXT: 'copyTextAttachmentsAsPlainText',
	CUSTOM_CSS: 'customCss',
	// PY_INTERPRETER_ENABLED: 'pyInterpreterEnabled',
	CUSTOM_JSON: 'customJson',
	DISABLE_AUTO_SCROLL: 'disableAutoScroll',
	// Developer
	DISABLE_REASONING_PARSING: 'disableReasoningParsing',
	DRY_ALLOWED_LENGTH: 'dry_allowed_length',
	DRY_BASE: 'dry_base',
	DRY_MULTIPLIER: 'dry_multiplier',
	DRY_PENALTY_LAST_N: 'dry_penalty_last_n',
	DYNATEMP_EXPONENT: 'dynatemp_exponent',
	DYNATEMP_RANGE: 'dynatemp_range',
	ENABLE_CONTINUE_GENERATION: 'enableContinueGeneration',
	EXCLUDE_REASONING_FROM_CONTEXT: 'excludeReasoningFromContext',
	FREQUENCY_PENALTY: 'frequency_penalty',
	FULL_HEIGHT_CODE_BLOCKS: 'fullHeightCodeBlocks',
	JS_SANDBOX_ENABLED: 'jsSandboxEnabled',
	MAX_IMAGE_RESOLUTION: 'maxImageMPixels',
	MAX_TOKENS: 'max_tokens',
	MCP_REQUEST_TIMEOUT_SECONDS: 'mcpRequestTimeoutSeconds',
	// MCP
	MCP_SERVERS: 'mcpServers',
	MENTION_SEARCH_MAX_DEPTH: 'mentionSearchMaxDepth',
	MIN_P: 'min_p',
	PASTE_LONG_TEXT_TO_FILE_LEN: 'pasteLongTextToFileLen',
	PDF_AS_IMAGE: 'pdfAsImage',
	// Performance
	PRE_ENCODE_CONVERSATION: 'preEncodeConversation',
	PRESENCE_PENALTY: 'presence_penalty',
	RENDER_THINKING_AS_MARKDOWN: 'renderThinkingAsMarkdown',
	RENDER_USER_CONTENT_AS_MARKDOWN: 'renderUserContentAsMarkdown',
	// Penalties
	REPEAT_LAST_N: 'repeat_last_n',
	REPEAT_PENALTY: 'repeat_penalty',
	SAMPLERS: 'samplers',
	SEND_ON_ENTER: 'sendOnEnter',
	SHOW_AGENTIC_TURN_STATS: 'showAgenticTurnStats',
	SHOW_BUILD_VERSION: 'showBuildVersion',
	SHOW_FULL_PATH_IN_MENTIONS: 'showFullPathInMentions',
	// Display
	SHOW_MESSAGE_STATS: 'showMessageStats',
	SHOW_MODEL_QUANTIZATION: 'showModelQuantization',
	SHOW_MODEL_TAGS: 'showModelTags',
	SHOW_MODEL_ORG_NAME_IN_TRIGGER: 'showModelOrgNameInTrigger',
	SHOW_RAW_MODEL_NAMES: 'showRawModelNames',
	SHOW_RAW_OUTPUT_SWITCH: 'showRawOutputSwitch',
	SHOW_SYSTEM_MESSAGE: 'showSystemMessage',
	SHOW_THOUGHT_IN_PROGRESS: 'showThoughtInProgress',
	SYMBOLIC_MATH_ENABLED: 'symbolicMathEnabled',
	SYSTEM_MESSAGE: 'systemMessage',
	// Sampling
	TEMPERATURE: 'temperature',
	// General
	THEME: 'theme',
	TITLE_GENERATION_PROMPT: 'titleGenerationPrompt',
	TITLE_GENERATION_USE_FIRST_LINE: 'titleGenerationUseFirstLine',
	TITLE_GENERATION_USE_LLM: 'titleGenerationUseLLM',
	TOP_K: 'top_k',
	TOP_P: 'top_p',
	TYP_P: 'typ_p',
	XTC_PROBABILITY: 'xtc_probability',
	XTC_THRESHOLD: 'xtc_threshold'
} as const;
