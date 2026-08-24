import { CLI_FLAGS } from './cli-flags.constants';
import { DEFAULT_MCP_CONFIG } from './mcp.constants';
import { SETTINGS_KEYS } from './settings-keys.constants';
import { TITLE_GENERATION } from './title-generation.constants';
import { FILE_GLOB_SEARCH_PICKERS } from './working-directory.constants';
import {
	Code,
	Database,
	Funnel,
	ListRestart,
	Monitor,
	Moon,
	PencilRuler,
	SlidersVertical,
	Sun
} from '@lucide/svelte';
import { SyncableParameterType } from '$lib/enums';
import { SettingsFieldType } from '$lib/enums/settings.enums';
import { ColorMode } from '$lib/enums/ui.enums';
import type {
	SettingsConfigValue,
	SettingsEntry,
	SettingsFieldConfig,
	SettingsSection,
	SettingsSectionEntry
} from '$lib/types';

/** Settings sections — slug is the routing identity, title is the display label. */
export const SETTINGS_SECTIONS = {
	AGENTIC: { slug: 'agentic', title: 'Agentic' },
	DEVELOPER: { slug: 'developer', title: 'Developer' },
	DISPLAY: { slug: 'display', title: 'Display' },
	GENERAL: { slug: 'general', title: 'General' },
	IMPORT_EXPORT: { slug: 'import-export', title: 'Import/Export' },
	SAMPLING_PENALTIES: { slug: 'sampling-penalties', title: 'Sampling & Penalties' },
	TOOLS: { slug: 'tools', title: 'Tools' }
} as const;

export const SETTINGS_SECTION_SLUGS = {
	AGENTIC: SETTINGS_SECTIONS.AGENTIC.slug,
	DEVELOPER: SETTINGS_SECTIONS.DEVELOPER.slug,
	DISPLAY: SETTINGS_SECTIONS.DISPLAY.slug,
	GENERAL: SETTINGS_SECTIONS.GENERAL.slug,
	IMPORT_EXPORT: SETTINGS_SECTIONS.IMPORT_EXPORT.slug,
	SAMPLING_PENALTIES: SETTINGS_SECTIONS.SAMPLING_PENALTIES.slug,
	TOOLS: SETTINGS_SECTIONS.TOOLS.slug
} as const;

export const SETTINGS_SECTION_TITLES = {
	AGENTIC: SETTINGS_SECTIONS.AGENTIC.title,
	DEVELOPER: SETTINGS_SECTIONS.DEVELOPER.title,
	DISPLAY: SETTINGS_SECTIONS.DISPLAY.title,
	GENERAL: SETTINGS_SECTIONS.GENERAL.title,
	IMPORT_EXPORT: SETTINGS_SECTIONS.IMPORT_EXPORT.title,
	SAMPLING_PENALTIES: SETTINGS_SECTIONS.SAMPLING_PENALTIES.title,
	TOOLS: SETTINGS_SECTIONS.TOOLS.title
} as const;

export const SETTINGS_REGISTRY: SettingsSectionEntry[] = [
	// General
	{
		icon: SlidersVertical,
		settings: [
			{
				defaultValue: ColorMode.SYSTEM,
				help: 'Choose the color theme for the interface. You can choose between System (follows your device settings), Light, or Dark.',
				key: SETTINGS_KEYS.THEME,
				label: 'Theme',
				options: [
					{ icon: Monitor, label: 'System', value: ColorMode.SYSTEM },
					{ icon: Sun, label: 'Light', value: ColorMode.LIGHT },
					{ icon: Moon, label: 'Dark', value: ColorMode.DARK }
				],
				type: SettingsFieldType.SELECT
			},
			{
				defaultValue: '',
				help: `Set the API Key if you are using <code> ${CLI_FLAGS.API_KEY} </code> option for the server.`,
				isPrivate: true,
				key: SETTINGS_KEYS.API_KEY,
				label: 'API Key',
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: '',
				help: 'The starting message that defines how model should behave.',
				key: SETTINGS_KEYS.SYSTEM_MESSAGE,
				label: 'System Message',
				type: SettingsFieldType.TEXTAREA
			},
			{
				defaultValue: true,
				help: 'Display the system message at the top of each conversation.',
				key: SETTINGS_KEYS.SHOW_SYSTEM_MESSAGE,
				label: 'Show system message',
				standaloneField: false,
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: 2500,
				help: 'On pasting long text, it will be converted to a file. You can control the file length by setting the value of this parameter. Value 0 means disable.',
				key: SETTINGS_KEYS.PASTE_LONG_TEXT_TO_FILE_LEN,
				label: 'Paste long text to file length',
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: true,
				help: 'Use Enter to send messages and Shift + Enter for new lines. When disabled, use Ctrl/Cmd + Enter.',
				key: SETTINGS_KEYS.SEND_ON_ENTER,
				label: 'Send message on Enter',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: 'Automatically show microphone button instead of send button when textarea is empty for models with audio modality support.',
				isExperimental: true,
				key: SETTINGS_KEYS.AUTO_MIC_ON_EMPTY,
				label: 'Show microphone on empty input',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: 'Enable "Continue" button for assistant messages, including reasoning models.',
				isExperimental: true,
				key: SETTINGS_KEYS.ENABLE_CONTINUE_GENERATION,
				label: 'Enable "Continue" button',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: true,
				help: 'Choose how conversation titles are generated. The first non-empty line uses a fast deterministic rule; the LLM option uses a model-generated title from the first message exchange.',
				key: SETTINGS_KEYS.TITLE_GENERATION_USE_FIRST_LINE,
				label: 'Conversation title',
				radioOptions: [
					{
						key: SETTINGS_KEYS.TITLE_GENERATION_USE_FIRST_LINE,
						label: 'Use first non-empty line for the conversation title',
						value: 'firstLine'
					},
					{
						isExperimental: true,
						key: SETTINGS_KEYS.TITLE_GENERATION_USE_LLM,
						label: 'Generate title with LLM',
						value: 'llm'
					}
				],
				type: SettingsFieldType.RADIO
			},
			{
				defaultValue: TITLE_GENERATION.DEFAULT_PROMPT,
				dependsOn: SETTINGS_KEYS.TITLE_GENERATION_USE_LLM,
				help: 'Optional template for the title generation prompt. Use {{USER}} for the user message and {{ASSISTANT}} for the assistant message.',
				key: SETTINGS_KEYS.TITLE_GENERATION_PROMPT,
				label: 'LLM title generation prompt',
				type: SettingsFieldType.TEXTAREA
			},
			{
				defaultValue: false,
				help: 'Counterpart of the conversation title radio; stored and synced without a dedicated UI field.',
				key: SETTINGS_KEYS.TITLE_GENERATION_USE_LLM,
				label: 'Generate title with LLM',
				standaloneField: false,
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: 'When copying a message with text attachments, combine them into a single plain text string instead of a special format that can be pasted back as attachments.',
				key: SETTINGS_KEYS.COPY_TEXT_ATTACHMENTS_AS_PLAIN_TEXT,
				label: 'Copy text attachments as plain text',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: 'Parse PDF as image instead of text. Automatically falls back to text processing for non-vision models.',
				key: SETTINGS_KEYS.PDF_AS_IMAGE,
				label: 'Parse PDF as image',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: 0,
				help: 'Images larger than this will be resized before sending to server. Set to 0 to disable.',
				key: SETTINGS_KEYS.MAX_IMAGE_RESOLUTION,
				label: 'Maximum image resolution (megapixels)',
				type: SettingsFieldType.INPUT
			}
		],
		slug: SETTINGS_SECTION_SLUGS.GENERAL,
		title: SETTINGS_SECTION_TITLES.GENERAL
	},
	// Display
	{
		icon: Monitor,
		settings: [
			{
				defaultValue: true,
				help: 'Display generation statistics (tokens/second, token count, duration) below each assistant message.',
				key: SETTINGS_KEYS.SHOW_MESSAGE_STATS,
				label: 'Show message generation statistics',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				dependsOn: SETTINGS_KEYS.SHOW_MESSAGE_STATS,
				help: 'Display per-turn statistics (tokens, duration) under each turn in agentic responses. Shown only when "Show message generation statistics" is enabled.',
				key: SETTINGS_KEYS.SHOW_AGENTIC_TURN_STATS,
				label: 'Show statistics for individual agentic turns',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: true,
				help: 'Expand thought process by default when generating messages.',
				key: SETTINGS_KEYS.SHOW_THOUGHT_IN_PROGRESS,
				label: 'Show thought in progress',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: 'Automatically expand tool call details while executing and keep them expanded after completion.',
				key: SETTINGS_KEYS.ALWAYS_SHOW_TOOL_CALL_CONTENT,
				label: 'Always show tool call content',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: true,
				help: 'Render user messages using markdown formatting in the chat. Turn this off to keep a message exactly as typed; @-mention badges show either way.',
				key: SETTINGS_KEYS.RENDER_USER_CONTENT_AS_MARKDOWN,
				label: 'Render user content as Markdown',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: true,
				help: 'Render the reasoning/thinking block content as formatted Markdown instead of plain text.',
				key: SETTINGS_KEYS.RENDER_THINKING_AS_MARKDOWN,
				label: 'Render thinking as Markdown',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: 'Always display code blocks at their full natural height, overriding any height limits.',
				key: SETTINGS_KEYS.FULL_HEIGHT_CODE_BLOCKS,
				label: 'Use full height code blocks',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: 'Disable automatic scrolling while messages stream so you can control the viewport position manually.',
				key: SETTINGS_KEYS.DISABLE_AUTO_SCROLL,
				label: 'Disable automatic scroll',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: 'Always keep the sidebar visible on desktop instead of auto-hiding it.',
				key: SETTINGS_KEYS.ALWAYS_SHOW_SIDEBAR_ON_DESKTOP,
				label: 'Always show sidebar on desktop',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: true,
				help: 'Show open chats as browser-style tabs above the conversation, one per open chat. When disabled, only one chat is shown at a time.',
				key: SETTINGS_KEYS.CONVERSATION_TABS,
				label: 'Conversation tabs',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: 'Display full raw model identifiers (e.g. "ggml-org/GLM-4.7-Flash-GGUF:Q8_0") instead of parsed names with badges.',
				key: SETTINGS_KEYS.SHOW_RAW_MODEL_NAMES,
				label: 'Show raw model names',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: true,
				help: 'Display quantization badges (e.g. Q8_0, Q4_K_M) next to model names throughout the interface.',
				key: SETTINGS_KEYS.SHOW_MODEL_QUANTIZATION,
				label: 'Show model quantization information',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: true,
				help: 'Display model tags (e.g. "vision", "reasoning") next to model names throughout the interface.',
				key: SETTINGS_KEYS.SHOW_MODEL_TAGS,
				label: 'Show model tags',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: 'Display the current build version in the bottom-right corner of the interface.',
				key: SETTINGS_KEYS.SHOW_BUILD_VERSION,
				label: 'Show build version information',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: 'Display the full file system path inside file and folder @-mention badges instead of just the file or folder name.',
				key: SETTINGS_KEYS.SHOW_FULL_PATH_IN_MENTIONS,
				label: 'Show full path in mentions',
				type: SettingsFieldType.CHECKBOX
			}
		],
		slug: SETTINGS_SECTION_SLUGS.DISPLAY,
		title: SETTINGS_SECTION_TITLES.DISPLAY
	},
	// MCP Servers (non-UI config object)
	{
		icon: PencilRuler,
		settings: [
			{
				defaultValue: '[]',
				help: 'Configure MCP servers as a JSON list. Use the form in the MCP Client settings section to edit.',
				key: SETTINGS_KEYS.MCP_SERVERS,
				label: 'MCP servers',
				standaloneField: false,
				type: SettingsFieldType.INPUT
			}
		],
		slug: SETTINGS_SECTION_SLUGS.TOOLS,
		title: SETTINGS_SECTION_TITLES.TOOLS
	},
	// Tools
	{
		icon: ListRestart,
		settings: [
			{
				defaultValue: 10,
				help: 'Maximum number of tool execution cycles before stopping (prevents infinite loops).',
				isPositiveInteger: true,
				key: SETTINGS_KEYS.AGENTIC_MAX_TURNS,
				label: 'Agentic turns',
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: DEFAULT_MCP_CONFIG.requestTimeoutSeconds,
				help: 'Timeout for individual MCP tool calls.',
				isPositiveInteger: true,
				key: SETTINGS_KEYS.MCP_REQUEST_TIMEOUT_SECONDS,
				label: 'MCP request timeout (seconds)',
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: FILE_GLOB_SEARCH_PICKERS.DEFAULT_SEARCH_DEPTH,
				help: 'How many directory levels below the working directory the @-mention file search descends. Larger values surface deeply nested files but take longer on large trees.',
				isPositiveInteger: true,
				key: SETTINGS_KEYS.MENTION_SEARCH_MAX_DEPTH,
				label: 'Mention search depth',
				max: FILE_GLOB_SEARCH_PICKERS.MAX_SEARCH_DEPTH,
				min: 1,
				placeholder: `${FILE_GLOB_SEARCH_PICKERS.DEFAULT_SEARCH_DEPTH}`,
				type: SettingsFieldType.INPUT
			}
		],
		slug: SETTINGS_SECTION_SLUGS.AGENTIC,
		title: SETTINGS_SECTION_TITLES.AGENTIC
	},
	// Import/Export
	{
		icon: Database,
		settings: [],
		slug: SETTINGS_SECTION_SLUGS.IMPORT_EXPORT,
		title: SETTINGS_SECTION_TITLES.IMPORT_EXPORT
	},
	// Sampling
	{
		icon: Funnel,
		settings: [
			{
				defaultValue: undefined,
				help: 'Controls the randomness of the generated text by affecting the probability distribution of the output tokens. Higher = more random, lower = more focused.',
				key: SETTINGS_KEYS.TEMPERATURE,
				label: 'Temperature',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.TEMPERATURE
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'Addon for the temperature sampler. The added value to the range of dynamic temperature, which adjusts probabilities by entropy of tokens.',
				key: SETTINGS_KEYS.DYNATEMP_RANGE,
				label: 'Dynamic temperature range',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.DYNATEMP_RANGE
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'Addon for the temperature sampler. Smoothes out the probability redistribution based on the most probable token.',
				key: SETTINGS_KEYS.DYNATEMP_EXPONENT,
				label: 'Dynamic temperature exponent',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.DYNATEMP_EXPONENT
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'Keeps only k top tokens.',
				key: SETTINGS_KEYS.TOP_K,
				label: 'Top K',
				sync: { paramType: SyncableParameterType.NUMBER, serverKey: SETTINGS_KEYS.TOP_K },
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'Limits tokens to those that together have a cumulative probability of at least p',
				key: SETTINGS_KEYS.TOP_P,
				label: 'Top P',
				sync: { paramType: SyncableParameterType.NUMBER, serverKey: SETTINGS_KEYS.TOP_P },
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'Limits tokens based on the minimum probability for a token to be considered, relative to the probability of the most likely token.',
				key: SETTINGS_KEYS.MIN_P,
				label: 'Min P',
				sync: { paramType: SyncableParameterType.NUMBER, serverKey: SETTINGS_KEYS.MIN_P },
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'XTC sampler cuts out top tokens; this parameter controls the chance of cutting tokens at all. 0 disables XTC.',
				key: SETTINGS_KEYS.XTC_PROBABILITY,
				label: 'XTC probability',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.XTC_PROBABILITY
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'XTC sampler cuts out top tokens; this parameter controls the token probability that is required to cut that token.',
				key: SETTINGS_KEYS.XTC_THRESHOLD,
				label: 'XTC threshold',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.XTC_THRESHOLD
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'Sorts and limits tokens based on the difference between log-probability and entropy.',
				key: SETTINGS_KEYS.TYP_P,
				label: 'Typical P',
				sync: { paramType: SyncableParameterType.NUMBER, serverKey: SETTINGS_KEYS.TYP_P },
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'The maximum number of token per output. Use -1 for infinite (no limit).',
				key: SETTINGS_KEYS.MAX_TOKENS,
				label: 'Max tokens',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.MAX_TOKENS
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: '',
				help: 'The order at which samplers are applied, in simplified way. Default is "top_k;typ_p;top_p;min_p;temperature": top_k->typ_p->top_p->min_p->temperature',
				key: SETTINGS_KEYS.SAMPLERS,
				label: 'Samplers',
				sync: { paramType: SyncableParameterType.STRING, serverKey: SETTINGS_KEYS.SAMPLERS },
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: false,
				help: 'Enable backend-based samplers. When enabled, supported samplers run on the accelerator backend for faster sampling.',
				key: SETTINGS_KEYS.BACKEND_SAMPLING,
				label: 'Backend sampling',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: undefined,
				help: 'Last n tokens to consider for penalizing repetition',
				key: SETTINGS_KEYS.REPEAT_LAST_N,
				label: 'Repeat last N',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.REPEAT_LAST_N
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'Controls the repetition of token sequences in the generated text',
				key: SETTINGS_KEYS.REPEAT_PENALTY,
				label: 'Repeat penalty',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.REPEAT_PENALTY
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'Limits tokens based on whether they appear in the output or not.',
				key: SETTINGS_KEYS.PRESENCE_PENALTY,
				label: 'Presence penalty',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.PRESENCE_PENALTY
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'Limits tokens based on how often they appear in the output.',
				key: SETTINGS_KEYS.FREQUENCY_PENALTY,
				label: 'Frequency penalty',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.FREQUENCY_PENALTY
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the DRY sampling multiplier.',
				key: SETTINGS_KEYS.DRY_MULTIPLIER,
				label: 'DRY multiplier',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.DRY_MULTIPLIER
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the DRY sampling base value.',
				key: SETTINGS_KEYS.DRY_BASE,
				label: 'DRY base',
				sync: { paramType: SyncableParameterType.NUMBER, serverKey: SETTINGS_KEYS.DRY_BASE },
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets the allowed length for DRY sampling.',
				key: SETTINGS_KEYS.DRY_ALLOWED_LENGTH,
				label: 'DRY allowed length',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.DRY_ALLOWED_LENGTH
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'DRY sampling reduces repetition in generated text even across long contexts. This parameter sets DRY penalty for the last n tokens.',
				key: SETTINGS_KEYS.DRY_PENALTY_LAST_N,
				label: 'DRY penalty last N',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.DRY_PENALTY_LAST_N
				},
				type: SettingsFieldType.INPUT
			}
		],
		slug: SETTINGS_SECTION_SLUGS.SAMPLING_PENALTIES,
		title: SETTINGS_SECTION_TITLES.SAMPLING_PENALTIES
	},
	// Developer
	{
		icon: Code,
		settings: [
			{
				defaultValue: false,
				help: 'After each response, re-submit the conversation to pre-fill the server KV cache. Makes the next turn faster since the prompt is already encoded while you read the response.',
				key: SETTINGS_KEYS.PRE_ENCODE_CONVERSATION,
				label: 'Pre-fill KV cache after response',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: 'Send reasoning_format=none so the server returns thinking tokens inline instead of extracting them into a separate field.',
				key: SETTINGS_KEYS.DISABLE_REASONING_PARSING,
				label: 'Disable reasoning content parsing',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: 'Strip thinking from previous messages before sending. When off, thinking is sent back via the reasoning_content field so the model sees its own chain-of-thought across turns.',
				key: SETTINGS_KEYS.EXCLUDE_REASONING_FROM_CONTEXT,
				label: 'Exclude reasoning from context',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: 'Show toggle button to display messages as plain text instead of Markdown-formatted content',
				key: SETTINGS_KEYS.SHOW_RAW_OUTPUT_SWITCH,
				label: 'Enable raw output toggle',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: 'Expose a run_javascript tool to the model. Code runs in a Web Worker inside a sandboxed iframe with an opaque origin, isolated from the WebUI and its API, with a hard timeout.',
				key: SETTINGS_KEYS.JS_SANDBOX_ENABLED,
				label: 'JavaScript sandbox tool',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				dependsOn: SETTINGS_KEYS.JS_SANDBOX_ENABLED,
				help: 'Pre-load nerdamer in the sandbox for symbolic computation: simplify, diff, integrate, solve, and more. Requires "JavaScript sandbox tool" to be enabled.',
				key: SETTINGS_KEYS.SYMBOLIC_MATH_ENABLED,
				label: 'Symbolic math (nerdamer)',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: '',
				help: 'Custom JSON parameters to send to the API. Must be valid JSON format.',
				key: SETTINGS_KEYS.CUSTOM_JSON,
				label: 'Custom JSON',
				type: SettingsFieldType.TEXTAREA
			},
			{
				defaultValue: '',
				help: 'CSS injected into the page at runtime. Set it here, or ship it server side via the --ui-config customCss field.',
				key: SETTINGS_KEYS.CUSTOM_CSS,
				label: 'Custom CSS',
				type: SettingsFieldType.TEXTAREA
			}
		],
		slug: SETTINGS_SECTION_SLUGS.DEVELOPER,
		title: SETTINGS_SECTION_TITLES.DEVELOPER
	}
];

function getAllSettings(): SettingsEntry[] {
	const result: SettingsEntry[] = [];

	for (const section of SETTINGS_REGISTRY) {
		result.push(...section.settings);
	}

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

/** Sidebar sections + field configs (as consumed by UI). */
function toSettingsSection(section: SettingsSectionEntry): SettingsSection {
	return {
		fields: section.settings
			.filter((s) => s.standaloneField !== false)
			.map((s) => ({
				dependsOn: s.dependsOn,
				help: s.help,
				isExperimental: s.isExperimental,
				isPositiveInteger: s.isPositiveInteger,
				isPrivate: s.isPrivate,
				key: s.key,
				label: s.label,
				max: s.max,
				min: s.min,
				options: s.options as SettingsFieldConfig['options'],
				placeholder: s.placeholder,
				radioOptions: s.radioOptions,
				type: s.type
			})),
		icon: section.icon,
		slug: section.slug,
		title: section.title
	};
}

/** Sidebar sections in custom display order (the registry array order). */
export const SETTINGS_CHAT_SECTIONS: SettingsSection[] = SETTINGS_REGISTRY.map(toSettingsSection);

/** INPUT-type settings whose value is a number. */
export const NUMERIC_FIELDS = getAllSettings()
	.filter((s) => s.type === SettingsFieldType.INPUT && typeof s.defaultValue !== 'string')
	.map((s) => s.key) as readonly string[];

/** Numeric fields clamped to >= 1 and rounded. */
export const POSITIVE_INTEGER_FIELDS = getAllSettings()
	.filter((s) => s.isPositiveInteger)
	.map((s) => s.key) as readonly string[];
