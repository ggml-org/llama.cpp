import { ColorMode } from '$lib/enums/ui';
import { Monitor, Moon, Sun } from '@lucide/svelte';

export const SETTING_CONFIG_DEFAULT: Record<string, string | number | boolean | undefined> = {
	// Note: in order not to introduce breaking changes, please keep the same data type (number, string, etc) if you want to change the default value.
	// Do not use nested objects, keep it single level. Prefix the key if you need to group them.
	apiKey: '',
	systemMessage: '',
	showSystemMessage: true,
	theme: ColorMode.SYSTEM,
	showThoughtInProgress: false,
	disableReasoningParsing: false,
	excludeReasoningFromContext: false,
	showRawOutputSwitch: false,
	keepStatsVisible: false,
	showMessageStats: true,
	askForTitleConfirmation: false,
	pasteLongTextToFileLen: 2500,
	copyTextAttachmentsAsPlainText: false,
	pdfAsImage: false,
	disableAutoScroll: false,
	renderUserContentAsMarkdown: false,
	alwaysShowSidebarOnDesktop: false,
	autoShowSidebarOnNewChat: true,
	sendOnEnter: true,
	autoMicOnEmpty: false,
	fullHeightCodeBlocks: false,
	showRawModelNames: false,
	mcpServers: '[]',
	mcpServerUsageStats: '{}', // JSON object: { [serverId]: usageCount }
	agenticMaxTurns: 10,
	agenticMaxToolPreviewLines: 25,
	showToolCallInProgress: false,
	alwaysShowAgenticTurns: false,
	// sampling params: empty means "use server default"
	// the server / preset is the source of truth
	// empty values are shown as placeholders from /props in the UI
	// and are NOT sent in API requests, letting the server decide
	samplers: '',
	backend_sampling: false,
	temperature: undefined,
	dynatemp_range: undefined,
	dynatemp_exponent: undefined,
	top_k: undefined,
	top_p: undefined,
	min_p: undefined,
	xtc_probability: undefined,
	xtc_threshold: undefined,
	typ_p: undefined,
	repeat_last_n: undefined,
	repeat_penalty: undefined,
	presence_penalty: undefined,
	frequency_penalty: undefined,
	dry_multiplier: undefined,
	dry_base: undefined,
	dry_allowed_length: undefined,
	dry_penalty_last_n: undefined,
	max_tokens: undefined,
	custom: '', // custom json-stringified object
	preEncodeConversation: false,
	// experimental features
	pyInterpreterEnabled: false,
	enableContinueGeneration: false
};

export const SETTING_CONFIG_INFO: Record<string, string> = {
	apiKey: 'Set the API Key if you are using <code>--api-key</code> option for the server.',
	systemMessage: 'The starting message that defines how model should behave.',
	showSystemMessage: 'Display the system message at the top of each conversation.',
	theme:
		'Choose the color theme for the interface. You can choose between System (follows your device settings), Light, or Dark.',
	pasteLongTextToFileLen:
		'On pasting long text, it will be converted to a file. You can control the file length by setting the value of this parameter. Value 0 means disable.',
	copyTextAttachmentsAsPlainText:
		'When copying a message with text attachments, combine them into a single plain text string instead of a special format that can be pasted back as attachments.',
	samplers:
		'The order at which samplers are applied, in simplified way. Default is "top_k;typ_p;top_p;min_p;temperature": top_k->typ_p->top_p->min_p->temperature',
	backend_sampling:
		'Enable backend-based samplers. When enabled, supported samplers run on the accelerator backend for faster sampling.',
	temperature:
		'Controls the randomness of the generated text by affecting the probability distribution of the output tokens. Higher = more random, lower = more focused.',
	dynatemp_range:
		'Addon for the temperature sampler. The added value to the range of dynamic temperature, which adjusts probabilities by entropy of tokens.',
	dynatemp_exponent:
		'Addon for the temperature sampler. Smoothes out the probability redistribution based on the most probable token.',
	top_k: 'Keeps only k top tokens.',
	top_p: 'Limits tokens to those that together have a cumulative probability of at least p',
	min_p:
		'Limits tokens based on the minimum probability for a token to be considered, relative to the probability of the most likely token.',
	xtc_probability:
		'XTC sampler cuts out top tokens; this parameter controls the chance of cutting tokens at all. 0 disables XTC.',
	xtc_threshold:
		'XTC sampler cuts out top tokens; this parameter controls the token probability that is required to cut that token.',
	typ_p: 'Sorts and limits tokens based on the difference between log-probability and entropy.',
	repeat_last_n: 'Last n tokens to consider for penalizing repetition',
	repeat_penalty: 'Controls the repetition of token sequences in the generated text',
	presence_penalty: 'Limits tokens based on whether they appear in the output or not.',
	frequency_penalty: 'Limits tokens based on how often they appear in the output.',
	dry_multiplier:
		'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the DRY sampling multiplier.',
	dry_base:
		'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the DRY sampling base value.',
	dry_allowed_length:
		'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the allowed length for DRY sampling.',
	dry_penalty_last_n:
		'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets DRY penalty for the last n tokens.',
	max_tokens: 'The maximum number of token per output. Use -1 for infinite (no limit).',
	custom: 'Custom JSON parameters to send to the API. Must be valid JSON format.',
	showThoughtInProgress: 'Expand thought process by default when generating messages.',
	disableReasoningParsing:
		'Send reasoning_format=none so the server returns thinking tokens inline instead of extracting them into a separate field.',
	excludeReasoningFromContext:
		'Strip thinking from previous messages before sending. When off, thinking is sent back via the reasoning_content field so the model sees its own chain-of-thought across turns.',
	showRawOutputSwitch:
		'Show toggle button to display messages as plain text instead of Markdown-formatted content',
	keepStatsVisible: 'Keep processing statistics visible after generation finishes.',
	showMessageStats:
		'Display generation statistics (tokens/second, token count, duration) below each assistant message.',
	askForTitleConfirmation:
		'Ask for confirmation before automatically changing conversation title when editing the first message.',
	pdfAsImage:
		'Parse PDF as image instead of text. Automatically falls back to text processing for non-vision models.',
	disableAutoScroll:
		'Disable automatic scrolling while messages stream so you can control the viewport position manually.',
	renderUserContentAsMarkdown: 'Render user messages using markdown formatting in the chat.',
	alwaysShowSidebarOnDesktop:
		'Always keep the sidebar visible on desktop instead of auto-hiding it.',
	autoShowSidebarOnNewChat:
		'Automatically show sidebar when starting a new chat. Disable to keep the sidebar hidden until you click on it.',
	sendOnEnter:
		'Use Enter to send messages and Shift + Enter for new lines. When disabled, use Ctrl/Cmd + Enter.',
	autoMicOnEmpty:
		'Automatically show microphone button instead of send button when textarea is empty for models with audio modality support.',
	fullHeightCodeBlocks:
		'Always display code blocks at their full natural height, overriding any height limits.',
	showRawModelNames:
		'Display full raw model identifiers (e.g. "ggml-org/GLM-4.7-Flash-GGUF:Q8_0") instead of parsed names with badges.',
	mcpServers:
		'Configure MCP servers as a JSON list. Use the form in the MCP Client settings section to edit.',
	mcpServerUsageStats:
		'Usage statistics for MCP servers. Tracks how many times tools from each server have been used.',
	agenticMaxTurns:
		'Maximum number of tool execution cycles before stopping (prevents infinite loops).',
	agenticMaxToolPreviewLines:
		'Number of lines shown in tool output previews (last N lines). Only these previews and the final LLM response persist after the agentic loop completes.',
	showToolCallInProgress:
		'Automatically expand tool call details while executing and keep them expanded after completion.',
	pyInterpreterEnabled:
		'Enable Python interpreter using Pyodide. Allows running Python code in markdown code blocks.',
	preEncodeConversation:
		'After each response, re-submit the conversation to pre-fill the server KV cache. Makes the next turn faster since the prompt is already encoded while you read the response.',
	enableContinueGeneration:
		'Enable "Continue" button for assistant messages. Currently works only with non-reasoning models.'
};

export const SETTINGS_COLOR_MODES_CONFIG = [
	{ value: ColorMode.SYSTEM, label: 'System', icon: Monitor },
	{ value: ColorMode.LIGHT, label: 'Light', icon: Sun },
	{ value: ColorMode.DARK, label: 'Dark', icon: Moon }
];
