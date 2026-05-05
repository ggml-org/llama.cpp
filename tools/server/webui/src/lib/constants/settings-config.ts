import { ColorMode } from '$lib/enums/ui';
import { Monitor, Moon, Sun } from '@lucide/svelte';
import { SETTINGS_KEYS } from './settings-keys';

export const SETTING_CONFIG_DEFAULT: Record<string, string | number | boolean | undefined> = {
	// Note: in order not to introduce breaking changes, please keep the same data type (number, string, etc) if you want to change the default value.
	// Do not use nested objects, keep it single level. Prefix the key if you need to group them.
	[SETTINGS_KEYS.API_KEY]: '',
	[SETTINGS_KEYS.SYSTEM_MESSAGE]: '',
	[SETTINGS_KEYS.SHOW_SYSTEM_MESSAGE]: true,
	[SETTINGS_KEYS.THEME]: ColorMode.SYSTEM,
	[SETTINGS_KEYS.SHOW_THOUGHT_IN_PROGRESS]: true,
	[SETTINGS_KEYS.DISABLE_REASONING_PARSING]: false,
	[SETTINGS_KEYS.EXCLUDE_REASONING_FROM_CONTEXT]: false,
	[SETTINGS_KEYS.SHOW_RAW_OUTPUT_SWITCH]: false,
	[SETTINGS_KEYS.KEEP_STATS_VISIBLE]: false,
	[SETTINGS_KEYS.SHOW_MESSAGE_STATS]: true,
	[SETTINGS_KEYS.ASK_FOR_TITLE_CONFIRMATION]: false,
	[SETTINGS_KEYS.TITLE_GENERATION_USE_FIRST_LINE]: false,
	[SETTINGS_KEYS.PASTE_LONG_TEXT_TO_FILE_LEN]: 2500,
	[SETTINGS_KEYS.COPY_TEXT_ATTACHMENTS_AS_PLAIN_TEXT]: false,
	[SETTINGS_KEYS.PDF_AS_IMAGE]: false,
	[SETTINGS_KEYS.DISABLE_AUTO_SCROLL]: false,
	[SETTINGS_KEYS.RENDER_USER_CONTENT_AS_MARKDOWN]: false,
	[SETTINGS_KEYS.ALWAYS_SHOW_SIDEBAR_ON_DESKTOP]: false,
	[SETTINGS_KEYS.AUTO_SHOW_SIDEBAR_ON_NEW_CHAT]: true,
	[SETTINGS_KEYS.SEND_ON_ENTER]: true,
	[SETTINGS_KEYS.AUTO_MIC_ON_EMPTY]: false,
	[SETTINGS_KEYS.FULL_HEIGHT_CODE_BLOCKS]: false,
	[SETTINGS_KEYS.SHOW_RAW_MODEL_NAMES]: false,
	[SETTINGS_KEYS.MCP_SERVERS]: '[]',
	[SETTINGS_KEYS.MCP_SERVER_USAGE_STATS]: '{}', // JSON object: { [serverId]: usageCount }
	[SETTINGS_KEYS.AGENTIC_MAX_TURNS]: 10,
	[SETTINGS_KEYS.AGENTIC_MAX_TOOL_PREVIEW_LINES]: 25,
	[SETTINGS_KEYS.SHOW_TOOL_CALL_IN_PROGRESS]: false,
	[SETTINGS_KEYS.ALWAYS_SHOW_AGENTIC_TURNS]: false,
	// sampling params: empty means "use server default"
	// the server / preset is the source of truth
	// empty values are shown as placeholders from /props in the UI
	// and are NOT sent in API requests, letting the server decide
	[SETTINGS_KEYS.SAMPLERS]: '',
	[SETTINGS_KEYS.BACKEND_SAMPLING]: false,
	[SETTINGS_KEYS.TEMPERATURE]: undefined,
	[SETTINGS_KEYS.DYNATEMP_RANGE]: undefined,
	[SETTINGS_KEYS.DYNATEMP_EXPONENT]: undefined,
	[SETTINGS_KEYS.TOP_K]: undefined,
	[SETTINGS_KEYS.TOP_P]: undefined,
	[SETTINGS_KEYS.MIN_P]: undefined,
	[SETTINGS_KEYS.XTC_PROBABILITY]: undefined,
	[SETTINGS_KEYS.XTC_THRESHOLD]: undefined,
	[SETTINGS_KEYS.TYP_P]: undefined,
	[SETTINGS_KEYS.REPEAT_LAST_N]: undefined,
	[SETTINGS_KEYS.REPEAT_PENALTY]: undefined,
	[SETTINGS_KEYS.PRESENCE_PENALTY]: undefined,
	[SETTINGS_KEYS.FREQUENCY_PENALTY]: undefined,
	[SETTINGS_KEYS.DRY_MULTIPLIER]: undefined,
	[SETTINGS_KEYS.DRY_BASE]: undefined,
	[SETTINGS_KEYS.DRY_ALLOWED_LENGTH]: undefined,
	[SETTINGS_KEYS.DRY_PENALTY_LAST_N]: undefined,
	[SETTINGS_KEYS.MAX_TOKENS]: undefined,
	[SETTINGS_KEYS.CUSTOM]: '', // custom json-stringified object
	[SETTINGS_KEYS.PRE_ENCODE_CONVERSATION]: false,
	// experimental features
	[SETTINGS_KEYS.PY_INTERPRETER_ENABLED]: false,
	[SETTINGS_KEYS.ENABLE_CONTINUE_GENERATION]: false
};

export const SETTING_CONFIG_INFO: Record<string, string> = {
	[SETTINGS_KEYS.API_KEY]:
		'Set the API Key if you are using <code>--api-key</code> option for the server.',
	[SETTINGS_KEYS.SYSTEM_MESSAGE]: 'The starting message that defines how model should behave.',
	[SETTINGS_KEYS.SHOW_SYSTEM_MESSAGE]:
		'Display the system message at the top of each conversation.',
	[SETTINGS_KEYS.THEME]:
		'Choose the color theme for the interface. You can choose between System (follows your device settings), Light, or Dark.',
	[SETTINGS_KEYS.PASTE_LONG_TEXT_TO_FILE_LEN]:
		'On pasting long text, it will be converted to a file. You can control the file length by setting the value of this parameter. Value 0 means disable.',
	[SETTINGS_KEYS.COPY_TEXT_ATTACHMENTS_AS_PLAIN_TEXT]:
		'When copying a message with text attachments, combine them into a single plain text string instead of a special format that can be pasted back as attachments.',
	[SETTINGS_KEYS.SAMPLERS]:
		'The order at which samplers are applied, in simplified way. Default is "top_k;typ_p;top_p;min_p;temperature": top_k->typ_p->top_p->min_p->temperature',
	[SETTINGS_KEYS.BACKEND_SAMPLING]:
		'Enable backend-based samplers. When enabled, supported samplers run on the accelerator backend for faster sampling.',
	[SETTINGS_KEYS.TEMPERATURE]:
		'Controls the randomness of the generated text by affecting the probability distribution of the output tokens. Higher = more random, lower = more focused.',
	[SETTINGS_KEYS.DYNATEMP_RANGE]:
		'Addon for the temperature sampler. The added value to the range of dynamic temperature, which adjusts probabilities by entropy of tokens.',
	[SETTINGS_KEYS.DYNATEMP_EXPONENT]:
		'Addon for the temperature sampler. Smoothes out the probability redistribution based on the most probable token.',
	[SETTINGS_KEYS.TOP_K]: 'Keeps only k top tokens.',
	[SETTINGS_KEYS.TOP_P]:
		'Limits tokens to those that together have a cumulative probability of at least p',
	[SETTINGS_KEYS.MIN_P]:
		'Limits tokens based on the minimum probability for a token to be considered, relative to the probability of the most likely token.',
	[SETTINGS_KEYS.XTC_PROBABILITY]:
		'XTC sampler cuts out top tokens; this parameter controls the chance of cutting tokens at all. 0 disables XTC.',
	[SETTINGS_KEYS.XTC_THRESHOLD]:
		'XTC sampler cuts out top tokens; this parameter controls the token probability that is required to cut that token.',
	[SETTINGS_KEYS.TYP_P]:
		'Sorts and limits tokens based on the difference between log-probability and entropy.',
	[SETTINGS_KEYS.REPEAT_LAST_N]: 'Last n tokens to consider for penalizing repetition',
	[SETTINGS_KEYS.REPEAT_PENALTY]:
		'Controls the repetition of token sequences in the generated text',
	[SETTINGS_KEYS.PRESENCE_PENALTY]:
		'Limits tokens based on whether they appear in the output or not.',
	[SETTINGS_KEYS.FREQUENCY_PENALTY]: 'Limits tokens based on how often they appear in the output.',
	[SETTINGS_KEYS.DRY_MULTIPLIER]:
		'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the DRY sampling multiplier.',
	[SETTINGS_KEYS.DRY_BASE]:
		'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the DRY sampling base value.',
	[SETTINGS_KEYS.DRY_ALLOWED_LENGTH]:
		'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the allowed length for DRY sampling.',
	[SETTINGS_KEYS.DRY_PENALTY_LAST_N]:
		'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets DRY penalty for the last n tokens.',
	[SETTINGS_KEYS.MAX_TOKENS]:
		'The maximum number of token per output. Use -1 for infinite (no limit).',
	[SETTINGS_KEYS.CUSTOM]: 'Custom JSON parameters to send to the API. Must be valid JSON format.',
	[SETTINGS_KEYS.SHOW_THOUGHT_IN_PROGRESS]:
		'Expand thought process by default when generating messages.',
	[SETTINGS_KEYS.DISABLE_REASONING_PARSING]:
		'Send reasoning_format=none so the server returns thinking tokens inline instead of extracting them into a separate field.',
	[SETTINGS_KEYS.EXCLUDE_REASONING_FROM_CONTEXT]:
		'Strip thinking from previous messages before sending. When off, thinking is sent back via the reasoning_content field so the model sees its own chain-of-thought across turns.',
	[SETTINGS_KEYS.SHOW_RAW_OUTPUT_SWITCH]:
		'Show toggle button to display messages as plain text instead of Markdown-formatted content',
	[SETTINGS_KEYS.KEEP_STATS_VISIBLE]:
		'Keep processing statistics visible after generation finishes.',
	[SETTINGS_KEYS.SHOW_MESSAGE_STATS]:
		'Display generation statistics (tokens/second, token count, duration) below each assistant message.',
	[SETTINGS_KEYS.ASK_FOR_TITLE_CONFIRMATION]:
		'Ask for confirmation before automatically changing conversation title when editing the first message.',
	[SETTINGS_KEYS.TITLE_GENERATION_USE_FIRST_LINE]:
		'Use only the first non-empty line of the prompt to generate the conversation title.',
	[SETTINGS_KEYS.PDF_AS_IMAGE]:
		'Parse PDF as image instead of text. Automatically falls back to text processing for non-vision models.',
	[SETTINGS_KEYS.DISABLE_AUTO_SCROLL]:
		'Disable automatic scrolling while messages stream so you can control the viewport position manually.',
	[SETTINGS_KEYS.RENDER_USER_CONTENT_AS_MARKDOWN]:
		'Render user messages using markdown formatting in the chat.',
	[SETTINGS_KEYS.ALWAYS_SHOW_SIDEBAR_ON_DESKTOP]:
		'Always keep the sidebar visible on desktop instead of auto-hiding it.',
	[SETTINGS_KEYS.AUTO_SHOW_SIDEBAR_ON_NEW_CHAT]:
		'Automatically show sidebar when starting a new chat. Disable to keep the sidebar hidden until you click on it.',
	[SETTINGS_KEYS.SEND_ON_ENTER]:
		'Use Enter to send messages and Shift + Enter for new lines. When disabled, use Ctrl/Cmd + Enter.',
	[SETTINGS_KEYS.AUTO_MIC_ON_EMPTY]:
		'Automatically show microphone button instead of send button when textarea is empty for models with audio modality support.',
	[SETTINGS_KEYS.FULL_HEIGHT_CODE_BLOCKS]:
		'Always display code blocks at their full natural height, overriding any height limits.',
	[SETTINGS_KEYS.SHOW_RAW_MODEL_NAMES]:
		'Display full raw model identifiers (e.g. "ggml-org/GLM-4.7-Flash-GGUF:Q8_0") instead of parsed names with badges.',
	[SETTINGS_KEYS.MCP_SERVERS]:
		'Configure MCP servers as a JSON list. Use the form in the MCP Client settings section to edit.',
	[SETTINGS_KEYS.MCP_SERVER_USAGE_STATS]:
		'Usage statistics for MCP servers. Tracks how many times tools from each server have been used.',
	[SETTINGS_KEYS.AGENTIC_MAX_TURNS]:
		'Maximum number of tool execution cycles before stopping (prevents infinite loops).',
	[SETTINGS_KEYS.AGENTIC_MAX_TOOL_PREVIEW_LINES]:
		'Number of lines shown in tool output previews (last N lines). Only these previews and the final LLM response persist after the agentic loop completes.',
	[SETTINGS_KEYS.SHOW_TOOL_CALL_IN_PROGRESS]:
		'Automatically expand tool call details while executing and keep them expanded after completion.',
	[SETTINGS_KEYS.PY_INTERPRETER_ENABLED]:
		'Enable Python interpreter using Pyodide. Allows running Python code in markdown code blocks.',
	[SETTINGS_KEYS.PRE_ENCODE_CONVERSATION]:
		'After each response, re-submit the conversation to pre-fill the server KV cache. Makes the next turn faster since the prompt is already encoded while you read the response.',
	[SETTINGS_KEYS.ENABLE_CONTINUE_GENERATION]:
		'Enable "Continue" button for assistant messages. Currently works only with non-reasoning models.'
};

export const SETTINGS_COLOR_MODES_CONFIG = [
	{ value: ColorMode.SYSTEM, label: 'System', icon: Monitor },
	{ value: ColorMode.LIGHT, label: 'Light', icon: Sun },
	{ value: ColorMode.DARK, label: 'Dark', icon: Moon }
];
