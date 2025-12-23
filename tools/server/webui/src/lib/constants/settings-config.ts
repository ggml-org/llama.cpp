// Ensure all built-in tools are registered before deriving defaults
import '$lib/services/tools';
import { getToolConfigDefaults, getToolSettingDefaults } from '$lib/services/tools/registry';

const BASE_SETTING_CONFIG_DEFAULT: Record<string, string | number | boolean> = {
	// Note: in order not to introduce breaking changes, please keep the same data type (number, string, etc) if you want to change the default value. Do not use null or undefined for default value.
	// Do not use nested objects, keep it single level. Prefix the key if you need to group them.
	apiKey: '',
	systemMessage: '',
	showSystemMessage: true,
	theme: 'system',
	showThoughtInProgress: false,
	showToolCalls: false,
	disableReasoningFormat: false,
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
	autoMicOnEmpty: false,
	// make sure these default values are in sync with `common.h`
	samplers: 'top_k;typ_p;top_p;min_p;temperature',
	temperature: 0.8,
	dynatemp_range: 0.0,
	dynatemp_exponent: 1.0,
	top_k: 40,
	top_p: 0.95,
	min_p: 0.05,
	xtc_probability: 0.0,
	xtc_threshold: 0.1,
	typ_p: 1.0,
	repeat_last_n: 64,
	repeat_penalty: 1.0,
	presence_penalty: 0.0,
	frequency_penalty: 0.0,
	dry_multiplier: 0.0,
	dry_base: 1.75,
	dry_allowed_length: 2,
	dry_penalty_last_n: -1,
	max_tokens: -1,
	custom: '', // custom json-stringified object
	// experimental features
	pyInterpreterEnabled: false,
	enableContinueGeneration: false
};

export const SETTING_CONFIG_DEFAULT: Record<string, string | number | boolean> = {
	...BASE_SETTING_CONFIG_DEFAULT,
	...getToolSettingDefaults(),
	...getToolConfigDefaults()
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
	showToolCalls:
		'Display tool call labels and payloads from Harmony-compatible delta.tool_calls data below assistant messages.',
	enableCalculatorTool:
		'Expose a simple calculator tool to the model. When the model calls it, the web UI evaluates the expression locally and resumes generation with the result.',
	enableCodeInterpreterTool:
		'Expose a JavaScript code interpreter to the model. Code runs in a sandboxed Worker and the result is sent back to the conversation.',
	disableReasoningFormat:
		'Show raw LLM output without backend parsing and frontend Markdown rendering to inspect streaming across different models.',
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
	autoMicOnEmpty:
		'Automatically show microphone button instead of send button when textarea is empty for models with audio modality support.',
	pyInterpreterEnabled:
		'Enable Python interpreter using Pyodide. Allows running Python code in markdown code blocks.',
	enableContinueGeneration:
		'Enable "Continue" button for assistant messages. Currently works only with non-reasoning models.'
};
