import { ColorMode } from '$lib/enums/ui';
import { SettingsFieldType } from '$lib/enums/settings';
import { SyncableParameterType } from '$lib/enums';
import {
	Funnel,
	AlertTriangle,
	Code,
	Monitor,
	ListRestart,
	Sliders,
	PencilRuler,
	Database,
	Monitor as MonitorIcon,
	Sun,
	Moon
} from '@lucide/svelte';
import type { Component } from 'svelte';
import type {
	SettingsConfigValue,
	SyncableParameter,
	SettingsEntry,
	SettingsSectionTitle,
	SettingsSectionEntry,
	SettingsSection
} from '$lib/types';
import { CLI_FLAGS, DEFAULT_MCP_CONFIG } from '$lib/constants';
import McpLogo from '$lib/components/app/mcp/McpLogo.svelte';
import { SETTINGS_KEYS } from './settings-keys';
import { ROUTES, SETTINGS_SECTION_SLUGS } from './routes';
import { TITLE_GENERATION } from './title-generation';

export const SETTINGS_SECTION_TITLES = {
	GENERAL: 'General',
	DISPLAY: 'Display',
	SAMPLING: 'Sampling',
	PENALTIES: 'Penalties',
	AGENTIC: 'Agentic',
	TOOLS: 'Tools',
	MCP: 'MCP',
	IMPORT_EXPORT: 'Import/Export',
	DEVELOPER: 'Developer'
} as const;

const STANDALONE_SECTIONS: { title: SettingsSectionTitle; slug: string; icon: Component }[] = [
	{ title: SETTINGS_SECTION_TITLES.TOOLS, slug: SETTINGS_SECTION_SLUGS.TOOLS, icon: PencilRuler },
	{
		title: SETTINGS_SECTION_TITLES.IMPORT_EXPORT,
		slug: SETTINGS_SECTION_SLUGS.IMPORT_EXPORT,
		icon: Database
	}
];

const COLOR_MODE_OPTIONS: Array<{ value: string; label: string; icon: Component }> = [
	{ value: ColorMode.SYSTEM, label: 'System', icon: MonitorIcon },
	{ value: ColorMode.LIGHT, label: 'Light', icon: Sun },
	{ value: ColorMode.DARK, label: 'Dark', icon: Moon }
];

const SETTINGS_REGISTRY: Record<string, SettingsSectionEntry> = {
	[SETTINGS_SECTION_SLUGS.GENERAL]: {
		title: SETTINGS_SECTION_TITLES.GENERAL,
		slug: SETTINGS_SECTION_SLUGS.GENERAL,
		icon: Sliders,
		settings: [
			{
				key: SETTINGS_KEYS.THEME,
				label: 'Theme',
				help: 'Choose the color theme for the interface. You can choose between System (follows your device settings), Light, or Dark.',
				defaultValue: ColorMode.SYSTEM,
				type: SettingsFieldType.SELECT,
				section: SETTINGS_SECTION_SLUGS.GENERAL,
				options: COLOR_MODE_OPTIONS,
				sync: { serverKey: SETTINGS_KEYS.THEME, paramType: SyncableParameterType.STRING }
			},
			{
				key: SETTINGS_KEYS.API_KEY,
				label: 'API Key',
				help: `Set the API Key if you are using <code> ${CLI_FLAGS.API_KEY} </code> option for the server.`,
				defaultValue: '',
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.GENERAL
			},
			{
				key: SETTINGS_KEYS.SYSTEM_MESSAGE,
				label: 'System Message',
				help: 'The starting message that defines how model should behave.',
				defaultValue: '',
				type: SettingsFieldType.TEXTAREA,
				section: SETTINGS_SECTION_SLUGS.GENERAL,
				sync: {
					serverKey: SETTINGS_KEYS.SYSTEM_MESSAGE,
					paramType: SyncableParameterType.STRING
				}
			},
			{
				key: SETTINGS_KEYS.PASTE_LONG_TEXT_TO_FILE_LEN,
				label: 'Paste long text to file length',
				help: 'On pasting long text, it will be converted to a file. You can control the file length by setting the value of this parameter. Value 0 means disable.',
				defaultValue: 2500,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.GENERAL,
				sync: {
					serverKey: SETTINGS_KEYS.PASTE_LONG_TEXT_TO_FILE_LEN,
					paramType: SyncableParameterType.NUMBER
				}
			},
			{
				key: SETTINGS_KEYS.SEND_ON_ENTER,
				label: 'Send message on Enter',
				help: 'Use Enter to send messages and Shift + Enter for new lines. When disabled, use Ctrl/Cmd + Enter.',
				defaultValue: true,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.GENERAL,
				sync: {
					serverKey: SETTINGS_KEYS.SEND_ON_ENTER,
					paramType: SyncableParameterType.BOOLEAN
				}
			},
			{
				key: SETTINGS_KEYS.COPY_TEXT_ATTACHMENTS_AS_PLAIN_TEXT,
				label: 'Copy text attachments as plain text',
				help: 'When copying a message with text attachments, combine them into a single plain text string instead of a special format that can be pasted back as attachments.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.GENERAL,
				sync: {
					serverKey: SETTINGS_KEYS.COPY_TEXT_ATTACHMENTS_AS_PLAIN_TEXT,
					paramType: SyncableParameterType.BOOLEAN
				}
			},
			{
				key: SETTINGS_KEYS.ENABLE_CONTINUE_GENERATION,
				label: 'Enable "Continue" button',
				help: 'Enable "Continue" button for assistant messages, including reasoning models.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.GENERAL,
				isExperimental: true,
				sync: {
					serverKey: SETTINGS_KEYS.ENABLE_CONTINUE_GENERATION,
					paramType: SyncableParameterType.BOOLEAN
				}
			},
			{
				key: SETTINGS_KEYS.PDF_AS_IMAGE,
				label: 'Parse PDF as image',
				help: 'Parse PDF as image instead of text. Automatically falls back to text processing for non-vision models.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.GENERAL,
				sync: {
					serverKey: SETTINGS_KEYS.PDF_AS_IMAGE,
					paramType: SyncableParameterType.BOOLEAN
				}
			},
			{
				key: SETTINGS_KEYS.ASK_FOR_TITLE_CONFIRMATION,
				label: 'Ask for confirmation before changing conversation title',
				help: 'Ask for confirmation before automatically changing conversation title when editing the first message.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.GENERAL,
				sync: {
					serverKey: SETTINGS_KEYS.ASK_FOR_TITLE_CONFIRMATION,
					paramType: SyncableParameterType.BOOLEAN
				}
			},
			{
				key: SETTINGS_KEYS.TITLE_GENERATION_USE_FIRST_LINE,
				label: 'Use first non-empty line for conversation title',
				help: 'Use only the first non-empty line of the prompt to generate the conversation title.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.GENERAL,
				sync: {
					serverKey: SETTINGS_KEYS.TITLE_GENERATION_USE_FIRST_LINE,
					paramType: SyncableParameterType.BOOLEAN
				}
			},
			{
				key: SETTINGS_KEYS.TITLE_GENERATION_USE_LLM,
				label: 'Use LLM to generate conversation title',
				help: 'Use the LLM to automatically generate conversation titles based on the first message exchange.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.GENERAL,
				isExperimental: true
			},
			{
				key: SETTINGS_KEYS.TITLE_GENERATION_PROMPT,
				label: 'LLM title generation prompt',
				help: 'Optional template for the title generation prompt. Use {{USER}} for the user message and {{ASSISTANT}} for the assistant message.',
				defaultValue: TITLE_GENERATION.DEFAULT_PROMPT,
				type: SettingsFieldType.TEXTAREA,
				section: SETTINGS_SECTION_SLUGS.GENERAL
			}
		]
	},
	[SETTINGS_SECTION_SLUGS.DISPLAY]: {
		title: SETTINGS_SECTION_TITLES.DISPLAY,
		slug: SETTINGS_SECTION_SLUGS.DISPLAY,
		icon: Monitor,
		settings: [
			{
				key: SETTINGS_KEYS.SHOW_MESSAGE_STATS,
				label: 'Show message generation statistics',
				help: 'Display generation statistics (tokens/second, token count, duration) below each assistant message.',
				defaultValue: true,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.DISPLAY,
				sync: {
					serverKey: SETTINGS_KEYS.SHOW_MESSAGE_STATS,
					paramType: SyncableParameterType.BOOLEAN
				}
			},
			{
				key: SETTINGS_KEYS.SHOW_THOUGHT_IN_PROGRESS,
				label: 'Show thought in progress',
				help: 'Expand thought process by default when generating messages.',
				defaultValue: true,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.DISPLAY,
				sync: {
					serverKey: SETTINGS_KEYS.SHOW_THOUGHT_IN_PROGRESS,
					paramType: SyncableParameterType.BOOLEAN
				}
			},
			{
				key: SETTINGS_KEYS.SHOW_TOOL_CALL_IN_PROGRESS,
				label: 'Show tool call in progress',
				help: 'Automatically expand tool call details while executing and keep them expanded after completion.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.DISPLAY,
				sync: {
					serverKey: SETTINGS_KEYS.SHOW_TOOL_CALL_IN_PROGRESS,
					paramType: SyncableParameterType.BOOLEAN
				}
			},
			{
				key: SETTINGS_KEYS.KEEP_STATS_VISIBLE,
				label: 'Keep stats visible after generation',
				help: 'Keep processing statistics visible after generation finishes.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.DISPLAY,
				sync: {
					serverKey: SETTINGS_KEYS.KEEP_STATS_VISIBLE,
					paramType: SyncableParameterType.BOOLEAN
				}
			},
			{
				key: SETTINGS_KEYS.AUTO_MIC_ON_EMPTY,
				label: 'Show microphone on empty input',
				help: 'Automatically show microphone button instead of send button when textarea is empty for models with audio modality support.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.DISPLAY,
				isExperimental: true,
				sync: {
					serverKey: SETTINGS_KEYS.AUTO_MIC_ON_EMPTY,
					paramType: SyncableParameterType.BOOLEAN
				}
			},
			{
				key: SETTINGS_KEYS.RENDER_USER_CONTENT_AS_MARKDOWN,
				label: 'Render user content as Markdown',
				help: 'Render user messages using markdown formatting in the chat.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.DISPLAY,
				sync: {
					serverKey: SETTINGS_KEYS.RENDER_USER_CONTENT_AS_MARKDOWN,
					paramType: SyncableParameterType.BOOLEAN
				}
			},
			{
				key: SETTINGS_KEYS.FULL_HEIGHT_CODE_BLOCKS,
				label: 'Use full height code blocks',
				help: 'Always display code blocks at their full natural height, overriding any height limits.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.DISPLAY,
				sync: {
					serverKey: SETTINGS_KEYS.FULL_HEIGHT_CODE_BLOCKS,
					paramType: SyncableParameterType.BOOLEAN
				}
			},
			{
				key: SETTINGS_KEYS.DISABLE_AUTO_SCROLL,
				label: 'Disable automatic scroll',
				help: 'Disable automatic scrolling while messages stream so you can control the viewport position manually.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.DISPLAY,
				sync: {
					serverKey: SETTINGS_KEYS.DISABLE_AUTO_SCROLL,
					paramType: SyncableParameterType.BOOLEAN
				}
			},
			{
				key: SETTINGS_KEYS.ALWAYS_SHOW_SIDEBAR_ON_DESKTOP,
				label: 'Always show sidebar on desktop',
				help: 'Always keep the sidebar visible on desktop instead of auto-hiding it.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.DISPLAY,
				sync: {
					serverKey: SETTINGS_KEYS.ALWAYS_SHOW_SIDEBAR_ON_DESKTOP,
					paramType: SyncableParameterType.BOOLEAN
				}
			},
			{
				key: SETTINGS_KEYS.SHOW_RAW_MODEL_NAMES,
				label: 'Show raw model names',
				help: 'Display full raw model identifiers (e.g. "ggml-org/GLM-4.7-Flash-GGUF:Q8_0") instead of parsed names with badges.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.DISPLAY,
				sync: {
					serverKey: SETTINGS_KEYS.SHOW_RAW_MODEL_NAMES,
					paramType: SyncableParameterType.BOOLEAN
				}
			},
			{
				key: SETTINGS_KEYS.ALWAYS_SHOW_AGENTIC_TURNS,
				label: 'Always show agentic turns in conversation',
				help: 'Always expand and display agentic loop turns in conversation messages.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.DISPLAY,
				sync: {
					serverKey: SETTINGS_KEYS.ALWAYS_SHOW_AGENTIC_TURNS,
					paramType: SyncableParameterType.BOOLEAN
				}
			}
		]
	},
	[SETTINGS_SECTION_SLUGS.SAMPLING]: {
		title: SETTINGS_SECTION_TITLES.SAMPLING,
		slug: SETTINGS_SECTION_SLUGS.SAMPLING,
		icon: Funnel,
		settings: [
			{
				key: SETTINGS_KEYS.TEMPERATURE,
				label: 'Temperature',
				help: 'Controls the randomness of the generated text by affecting the probability distribution of the output tokens. Higher = more random, lower = more focused.',
				defaultValue: undefined,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.SAMPLING,
				sync: {
					serverKey: SETTINGS_KEYS.TEMPERATURE,
					paramType: SyncableParameterType.NUMBER
				}
			},
			{
				key: SETTINGS_KEYS.DYNATEMP_RANGE,
				label: 'Dynamic temperature range',
				help: 'Addon for the temperature sampler. The added value to the range of dynamic temperature, which adjusts probabilities by entropy of tokens.',
				defaultValue: undefined,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.SAMPLING,
				sync: {
					serverKey: SETTINGS_KEYS.DYNATEMP_RANGE,
					paramType: SyncableParameterType.NUMBER
				}
			},
			{
				key: SETTINGS_KEYS.DYNATEMP_EXPONENT,
				label: 'Dynamic temperature exponent',
				help: 'Addon for the temperature sampler. Smoothes out the probability redistribution based on the most probable token.',
				defaultValue: undefined,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.SAMPLING,
				sync: {
					serverKey: SETTINGS_KEYS.DYNATEMP_EXPONENT,
					paramType: SyncableParameterType.NUMBER
				}
			},
			{
				key: SETTINGS_KEYS.TOP_K,
				label: 'Top K',
				help: 'Keeps only k top tokens.',
				defaultValue: undefined,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.SAMPLING,
				sync: { serverKey: SETTINGS_KEYS.TOP_K, paramType: SyncableParameterType.NUMBER }
			},
			{
				key: SETTINGS_KEYS.TOP_P,
				label: 'Top P',
				help: 'Limits tokens to those that together have a cumulative probability of at least p',
				defaultValue: undefined,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.SAMPLING,
				sync: { serverKey: SETTINGS_KEYS.TOP_P, paramType: SyncableParameterType.NUMBER }
			},
			{
				key: SETTINGS_KEYS.MIN_P,
				label: 'Min P',
				help: 'Limits tokens based on the minimum probability for a token to be considered, relative to the probability of the most likely token.',
				defaultValue: undefined,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.SAMPLING,
				sync: { serverKey: SETTINGS_KEYS.MIN_P, paramType: SyncableParameterType.NUMBER }
			},
			{
				key: SETTINGS_KEYS.XTC_PROBABILITY,
				label: 'XTC probability',
				help: 'XTC sampler cuts out top tokens; this parameter controls the chance of cutting tokens at all. 0 disables XTC.',
				defaultValue: undefined,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.SAMPLING,
				sync: {
					serverKey: SETTINGS_KEYS.XTC_PROBABILITY,
					paramType: SyncableParameterType.NUMBER
				}
			},
			{
				key: SETTINGS_KEYS.XTC_THRESHOLD,
				label: 'XTC threshold',
				help: 'XTC sampler cuts out top tokens; this parameter controls the token probability that is required to cut that token.',
				defaultValue: undefined,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.SAMPLING,
				sync: {
					serverKey: SETTINGS_KEYS.XTC_THRESHOLD,
					paramType: SyncableParameterType.NUMBER
				}
			},
			{
				key: SETTINGS_KEYS.TYP_P,
				label: 'Typical P',
				help: 'Sorts and limits tokens based on the difference between log-probability and entropy.',
				defaultValue: undefined,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.SAMPLING,
				sync: { serverKey: SETTINGS_KEYS.TYP_P, paramType: SyncableParameterType.NUMBER }
			},
			{
				key: SETTINGS_KEYS.MAX_TOKENS,
				label: 'Max tokens',
				help: 'The maximum number of token per output. Use -1 for infinite (no limit).',
				defaultValue: undefined,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.SAMPLING,
				sync: {
					serverKey: SETTINGS_KEYS.MAX_TOKENS,
					paramType: SyncableParameterType.NUMBER
				}
			},
			{
				key: SETTINGS_KEYS.SAMPLERS,
				label: 'Samplers',
				help: 'The order at which samplers are applied, in simplified way. Default is "top_k;typ_p;top_p;min_p;temperature": top_k->typ_p->top_p->min_p->temperature',
				defaultValue: '',
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.SAMPLING,
				sync: { serverKey: SETTINGS_KEYS.SAMPLERS, paramType: SyncableParameterType.STRING }
			},
			{
				key: SETTINGS_KEYS.BACKEND_SAMPLING,
				label: 'Backend sampling',
				help: 'Enable backend-based samplers. When enabled, supported samplers run on the accelerator backend for faster sampling.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.SAMPLING,
				sync: {
					serverKey: SETTINGS_KEYS.BACKEND_SAMPLING,
					paramType: SyncableParameterType.BOOLEAN
				}
			}
		]
	},
	[SETTINGS_SECTION_SLUGS.PENALTIES]: {
		title: SETTINGS_SECTION_TITLES.PENALTIES,
		slug: SETTINGS_SECTION_SLUGS.PENALTIES,
		icon: AlertTriangle,
		settings: [
			{
				key: SETTINGS_KEYS.REPEAT_LAST_N,
				label: 'Repeat last N',
				help: 'Last n tokens to consider for penalizing repetition',
				defaultValue: undefined,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.PENALTIES,
				sync: {
					serverKey: SETTINGS_KEYS.REPEAT_LAST_N,
					paramType: SyncableParameterType.NUMBER
				}
			},
			{
				key: SETTINGS_KEYS.REPEAT_PENALTY,
				label: 'Repeat penalty',
				help: 'Controls the repetition of token sequences in the generated text',
				defaultValue: undefined,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.PENALTIES,
				sync: {
					serverKey: SETTINGS_KEYS.REPEAT_PENALTY,
					paramType: SyncableParameterType.NUMBER
				}
			},
			{
				key: SETTINGS_KEYS.PRESENCE_PENALTY,
				label: 'Presence penalty',
				help: 'Limits tokens based on whether they appear in the output or not.',
				defaultValue: undefined,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.PENALTIES,
				sync: {
					serverKey: SETTINGS_KEYS.PRESENCE_PENALTY,
					paramType: SyncableParameterType.NUMBER
				}
			},
			{
				key: SETTINGS_KEYS.FREQUENCY_PENALTY,
				label: 'Frequency penalty',
				help: 'Limits tokens based on how often they appear in the output.',
				defaultValue: undefined,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.PENALTIES,
				sync: {
					serverKey: SETTINGS_KEYS.FREQUENCY_PENALTY,
					paramType: SyncableParameterType.NUMBER
				}
			},
			{
				key: SETTINGS_KEYS.DRY_MULTIPLIER,
				label: 'DRY multiplier',
				help: 'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the DRY sampling multiplier.',
				defaultValue: undefined,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.PENALTIES,
				sync: {
					serverKey: SETTINGS_KEYS.DRY_MULTIPLIER,
					paramType: SyncableParameterType.NUMBER
				}
			},
			{
				key: SETTINGS_KEYS.DRY_BASE,
				label: 'DRY base',
				help: 'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the DRY sampling base value.',
				defaultValue: undefined,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.PENALTIES,
				sync: { serverKey: SETTINGS_KEYS.DRY_BASE, paramType: SyncableParameterType.NUMBER }
			},
			{
				key: SETTINGS_KEYS.DRY_ALLOWED_LENGTH,
				label: 'DRY allowed length',
				help: 'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the allowed length for DRY sampling.',
				defaultValue: undefined,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.PENALTIES,
				sync: {
					serverKey: SETTINGS_KEYS.DRY_ALLOWED_LENGTH,
					paramType: SyncableParameterType.NUMBER
				}
			},
			{
				key: SETTINGS_KEYS.DRY_PENALTY_LAST_N,
				label: 'DRY penalty last N',
				help: 'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets DRY penalty for the last n tokens.',
				defaultValue: undefined,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.PENALTIES,
				sync: {
					serverKey: SETTINGS_KEYS.DRY_PENALTY_LAST_N,
					paramType: SyncableParameterType.NUMBER
				}
			}
		]
	},
	[SETTINGS_SECTION_SLUGS.AGENTIC]: {
		title: SETTINGS_SECTION_TITLES.AGENTIC,
		slug: SETTINGS_SECTION_SLUGS.AGENTIC,
		icon: ListRestart,
		settings: [
			{
				key: SETTINGS_KEYS.AGENTIC_MAX_TURNS,
				label: 'Agentic turns',
				help: 'Maximum number of tool execution cycles before stopping (prevents infinite loops).',
				defaultValue: 10,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.AGENTIC,
				isPositiveInteger: true,
				sync: {
					serverKey: SETTINGS_KEYS.AGENTIC_MAX_TURNS,
					paramType: SyncableParameterType.NUMBER
				}
			},
			{
				key: SETTINGS_KEYS.AGENTIC_MAX_TOOL_PREVIEW_LINES,
				label: 'Max lines per tool preview',
				help: 'Number of lines shown in tool output previews (last N lines). Only these previews and the final LLM response persist after the agentic loop completes.',
				defaultValue: 25,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.AGENTIC,
				isPositiveInteger: true,
				sync: {
					serverKey: SETTINGS_KEYS.AGENTIC_MAX_TOOL_PREVIEW_LINES,
					paramType: SyncableParameterType.NUMBER
				}
			}
		]
	},
	[SETTINGS_SECTION_SLUGS.DEVELOPER]: {
		title: SETTINGS_SECTION_TITLES.DEVELOPER,
		slug: SETTINGS_SECTION_SLUGS.DEVELOPER,
		icon: Code,
		settings: [
			{
				key: SETTINGS_KEYS.PRE_ENCODE_CONVERSATION,
				label: 'Pre-fill KV cache after response',
				help: 'After each response, re-submit the conversation to pre-fill the server KV cache. Makes the next turn faster since the prompt is already encoded while you read the response.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.DEVELOPER
			},
			{
				key: SETTINGS_KEYS.DISABLE_REASONING_PARSING,
				label: 'Disable reasoning content parsing',
				help: 'Send reasoning_format=none so the server returns thinking tokens inline instead of extracting them into a separate field.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.DEVELOPER
			},
			{
				key: SETTINGS_KEYS.EXCLUDE_REASONING_FROM_CONTEXT,
				label: 'Exclude reasoning from context',
				help: 'Strip thinking from previous messages before sending. When off, thinking is sent back via the reasoning_content field so the model sees its own chain-of-thought across turns.',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.DEVELOPER,
				sync: {
					serverKey: SETTINGS_KEYS.EXCLUDE_REASONING_FROM_CONTEXT,
					paramType: SyncableParameterType.BOOLEAN
				}
			},
			{
				key: SETTINGS_KEYS.SHOW_RAW_OUTPUT_SWITCH,
				label: 'Enable raw output toggle',
				help: 'Show toggle button to display messages as plain text instead of Markdown-formatted content',
				defaultValue: false,
				type: SettingsFieldType.CHECKBOX,
				section: SETTINGS_SECTION_SLUGS.DEVELOPER,
				sync: {
					serverKey: SETTINGS_KEYS.SHOW_RAW_OUTPUT_SWITCH,
					paramType: SyncableParameterType.BOOLEAN
				}
			},
			{
				key: SETTINGS_KEYS.CUSTOM,
				label: 'Custom JSON',
				help: 'Custom JSON parameters to send to the API. Must be valid JSON format.',
				defaultValue: '',
				type: SettingsFieldType.TEXTAREA,
				section: SETTINGS_SECTION_SLUGS.DEVELOPER
			}
		]
	},
	[SETTINGS_SECTION_SLUGS.MCP]: {
		title: SETTINGS_SECTION_TITLES.MCP,
		slug: SETTINGS_SECTION_SLUGS.MCP,
		icon: McpLogo,
		settings: [
			{
				key: SETTINGS_KEYS.MCP_REQUEST_TIMEOUT_SECONDS,
				label: 'Request timeout (seconds)',
				help: 'Default timeout for individual MCP tool calls. Can be overridden per server.',
				defaultValue: DEFAULT_MCP_CONFIG.requestTimeoutSeconds,
				type: SettingsFieldType.INPUT,
				section: SETTINGS_SECTION_SLUGS.MCP,
				isPositiveInteger: true
			}
		]
	}
} as const;

const NON_UI_SETTINGS: SettingsEntry[] = [
	{
		key: SETTINGS_KEYS.SHOW_SYSTEM_MESSAGE,
		label: 'Show system message',
		help: 'Display the system message at the top of each conversation.',
		defaultValue: true,
		type: SettingsFieldType.CHECKBOX,
		sync: {
			serverKey: SETTINGS_KEYS.SHOW_SYSTEM_MESSAGE,
			paramType: SyncableParameterType.BOOLEAN
		}
	},
	{
		key: SETTINGS_KEYS.MCP_SERVERS,
		label: 'MCP servers',
		help: 'Configure MCP servers as a JSON list. Use the form in the MCP Client settings section to edit.',
		defaultValue: '[]',
		type: SettingsFieldType.INPUT,
		sync: { serverKey: SETTINGS_KEYS.MCP_SERVERS, paramType: SyncableParameterType.STRING }
	}
	// {
	// 	key: SETTINGS_KEYS.PY_INTERPRETER_ENABLED,
	// 	label: 'Python interpreter enabled',
	// 	help: 'Enable Python interpreter using Pyodide. Allows running Python code in markdown code blocks.',
	// 	defaultValue: false,
	// 	type: SettingsFieldType.CHECKBOX,
	// 	isExperimental: true,
	// 	sync: { serverKey: SETTINGS_KEYS.PY_INTERPRETER_ENABLED, paramType: SyncableParameterType.BOOLEAN }
	// }
];

function getAllSettings(): SettingsEntry[] {
	const result: SettingsEntry[] = [];
	for (const section of Object.values(SETTINGS_REGISTRY)) {
		result.push(...section.settings);
	}
	result.push(...NON_UI_SETTINGS);
	return result;
}

/** Flat config object stored in localStorage. */
export const SETTING_CONFIG_DEFAULT: Record<string, SettingsConfigValue> = Object.fromEntries(
	getAllSettings().map((s) => [s.key, s.defaultValue])
) as Record<string, SettingsConfigValue>;

/** Help text for every setting (including non-UI). */
export const SETTING_CONFIG_INFO: Record<string, string> = Object.fromEntries(
	getAllSettings().map((s) => [s.key, s.help])
) as Record<string, string>;

/** Theme select options. */
export const SETTINGS_COLOR_MODES_CONFIG = COLOR_MODE_OPTIONS;

export type { SettingsSectionTitle } from '$lib/types';
export type { SettingsSection } from '$lib/types';

/** Sidebar sections + field configs (as consumed by UI). */
export const SETTINGS_CHAT_SECTIONS: SettingsSection[] = [
	...Object.values(SETTINGS_REGISTRY).map((section) => ({
		title: section.title,
		slug: section.slug,
		icon: section.icon,
		fields: section.settings.map((s) => ({
			key: s.key,
			label: s.label,
			type: s.type,
			isExperimental: s.isExperimental,
			isPositiveInteger: s.isPositiveInteger,
			help: s.help,
			options: s.options
		}))
	})),
	...STANDALONE_SECTIONS
];

/** INPUT-type settings whose value is a number. */
export const NUMERIC_FIELDS = getAllSettings()
	.filter((s) => s.type === SettingsFieldType.INPUT && typeof s.defaultValue !== 'string')
	.map((s) => s.key) as readonly string[];

/** Numeric fields clamped to ≥ 1 and rounded. */
export const POSITIVE_INTEGER_FIELDS = getAllSettings()
	.filter((s) => s.isPositiveInteger)
	.map((s) => s.key) as readonly string[];

/** Derived for the parameter sync service. */
export const SYNCABLE_PARAMETERS: SyncableParameter[] = getAllSettings()
	.filter((s) => s.sync !== undefined)
	.map((s) => ({
		key: s.key,
		serverKey: s.sync!.serverKey,
		type: s.sync!.paramType,
		canSync: true
	}));

export const SETTINGS_FALLBACK_EXIT_ROUTE = ROUTES.START;

export { SETTINGS_KEYS } from './settings-keys';
