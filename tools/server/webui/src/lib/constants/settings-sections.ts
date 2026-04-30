/**
 * Settings section titles constants for ChatSettings component.
 *
 * These titles define the navigation sections in the settings dialog.
 * Used for both sidebar navigation and mobile horizontal scroll menu.
 */
export const SETTINGS_SECTION_TITLES = {
	GENERAL: 'General',
	DISPLAY: 'Display',
	SAMPLING: 'Sampling',
	PENALTIES: 'Penalties',
	IMPORT_EXPORT: 'Import/Export',
	AGENTIC: 'Agentic',
	TOOLS: 'Tools',
	MCP: 'MCP',
	DEVELOPER: 'Developer'
} as const;

/** Type for settings section titles */
export type SettingsSectionTitle =
	(typeof SETTINGS_SECTION_TITLES)[keyof typeof SETTINGS_SECTION_TITLES];

import {
	Funnel,
	AlertTriangle,
	Code,
	Monitor,
	ListRestart,
	Sliders,
	PencilRuler
} from '@lucide/svelte';
import { SettingsFieldType } from '$lib/enums/settings';
import { SETTINGS_COLOR_MODES_CONFIG } from '$lib/constants/settings-config';
import { SETTINGS_KEYS } from '$lib/constants/settings-keys';
import type { Component } from 'svelte';

export interface SettingsSection {
	fields?: SettingsFieldConfig[];
	icon: Component;
	slug: string;
	title: SettingsSectionTitle;
}

export const SETTINGS_CHAT_SECTIONS: SettingsSection[] = [
	{
		title: SETTINGS_SECTION_TITLES.GENERAL,
		slug: 'general',
		icon: Sliders,
		fields: [
			{
				key: SETTINGS_KEYS.THEME,
				label: 'Theme',
				type: SettingsFieldType.SELECT,
				options: SETTINGS_COLOR_MODES_CONFIG
			},
			{ key: SETTINGS_KEYS.API_KEY, label: 'API Key', type: SettingsFieldType.INPUT },
			{
				key: SETTINGS_KEYS.SYSTEM_MESSAGE,
				label: 'System Message',
				type: SettingsFieldType.TEXTAREA
			},
			{
				key: SETTINGS_KEYS.PASTE_LONG_TEXT_TO_FILE_LEN,
				label: 'Paste long text to file length',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.COPY_TEXT_ATTACHMENTS_AS_PLAIN_TEXT,
				label: 'Copy text attachments as plain text',
				type: SettingsFieldType.CHECKBOX
			},
			{
				key: SETTINGS_KEYS.ENABLE_CONTINUE_GENERATION,
				label: 'Enable "Continue" button',
				type: SettingsFieldType.CHECKBOX,
				isExperimental: true
			},
			{
				key: SETTINGS_KEYS.PDF_AS_IMAGE,
				label: 'Parse PDF as image',
				type: SettingsFieldType.CHECKBOX
			},
			{
				key: SETTINGS_KEYS.ASK_FOR_TITLE_CONFIRMATION,
				label: 'Ask for confirmation before changing conversation title',
				type: SettingsFieldType.CHECKBOX
			}
		]
	},
	{
		title: SETTINGS_SECTION_TITLES.DISPLAY,
		slug: 'display',
		icon: Monitor,
		fields: [
			{
				key: SETTINGS_KEYS.SHOW_MESSAGE_STATS,
				label: 'Show message generation statistics',
				type: SettingsFieldType.CHECKBOX
			},
			{
				key: SETTINGS_KEYS.SHOW_THOUGHT_IN_PROGRESS,
				label: 'Show thought in progress',
				type: SettingsFieldType.CHECKBOX
			},
			{
				key: SETTINGS_KEYS.SHOW_TOOL_CALL_IN_PROGRESS,
				label: 'Show tool call in progress',
				type: SettingsFieldType.CHECKBOX
			},
			{
				key: SETTINGS_KEYS.KEEP_STATS_VISIBLE,
				label: 'Keep stats visible after generation',
				type: SettingsFieldType.CHECKBOX
			},
			{
				key: SETTINGS_KEYS.AUTO_MIC_ON_EMPTY,
				label: 'Show microphone on empty input',
				type: SettingsFieldType.CHECKBOX,
				isExperimental: true
			},
			{
				key: SETTINGS_KEYS.RENDER_USER_CONTENT_AS_MARKDOWN,
				label: 'Render user content as Markdown',
				type: SettingsFieldType.CHECKBOX
			},
			{
				key: SETTINGS_KEYS.FULL_HEIGHT_CODE_BLOCKS,
				label: 'Use full height code blocks',
				type: SettingsFieldType.CHECKBOX
			},
			{
				key: SETTINGS_KEYS.DISABLE_AUTO_SCROLL,
				label: 'Disable automatic scroll',
				type: SettingsFieldType.CHECKBOX
			},
			{
				key: SETTINGS_KEYS.ALWAYS_SHOW_SIDEBAR_ON_DESKTOP,
				label: 'Always show sidebar on desktop',
				type: SettingsFieldType.CHECKBOX
			},
			{
				key: SETTINGS_KEYS.SHOW_RAW_MODEL_NAMES,
				label: 'Show raw model names',
				type: SettingsFieldType.CHECKBOX
			},
			{
				key: SETTINGS_KEYS.ALWAYS_SHOW_AGENTIC_TURNS,
				label: 'Always show agentic turns in conversation',
				type: SettingsFieldType.CHECKBOX
			}
		]
	},
	{
		title: SETTINGS_SECTION_TITLES.SAMPLING,
		slug: 'sampling',
		icon: Funnel,
		fields: [
			{
				key: SETTINGS_KEYS.TEMPERATURE,
				label: 'Temperature',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.DYNATEMP_RANGE,
				label: 'Dynamic temperature range',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.DYNATEMP_EXPONENT,
				label: 'Dynamic temperature exponent',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.TOP_K,
				label: 'Top K',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.TOP_P,
				label: 'Top P',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.MIN_P,
				label: 'Min P',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.XTC_PROBABILITY,
				label: 'XTC probability',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.XTC_THRESHOLD,
				label: 'XTC threshold',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.TYP_P,
				label: 'Typical P',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.MAX_TOKENS,
				label: 'Max tokens',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.SAMPLERS,
				label: 'Samplers',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.BACKEND_SAMPLING,
				label: 'Backend sampling',
				type: SettingsFieldType.CHECKBOX
			}
		]
	},
	{
		title: SETTINGS_SECTION_TITLES.PENALTIES,
		slug: 'penalties',
		icon: AlertTriangle,
		fields: [
			{
				key: SETTINGS_KEYS.REPEAT_LAST_N,
				label: 'Repeat last N',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.REPEAT_PENALTY,
				label: 'Repeat penalty',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.PRESENCE_PENALTY,
				label: 'Presence penalty',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.FREQUENCY_PENALTY,
				label: 'Frequency penalty',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.DRY_MULTIPLIER,
				label: 'DRY multiplier',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.DRY_BASE,
				label: 'DRY base',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.DRY_ALLOWED_LENGTH,
				label: 'DRY allowed length',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.DRY_PENALTY_LAST_N,
				label: 'DRY penalty last N',
				type: SettingsFieldType.INPUT
			}
		]
	},
	{
		title: SETTINGS_SECTION_TITLES.AGENTIC,
		slug: 'agentic',
		icon: ListRestart,
		fields: [
			{
				key: SETTINGS_KEYS.AGENTIC_MAX_TURNS,
				label: 'Agentic turns',
				type: SettingsFieldType.INPUT
			},
			{
				key: SETTINGS_KEYS.AGENTIC_MAX_TOOL_PREVIEW_LINES,
				label: 'Max lines per tool preview',
				type: SettingsFieldType.INPUT
			}
		]
	},
	{
		title: SETTINGS_SECTION_TITLES.TOOLS,
		slug: 'tools',
		icon: PencilRuler
	},
	{
		title: SETTINGS_SECTION_TITLES.DEVELOPER,
		slug: 'developer',
		icon: Code,
		fields: [
			{
				key: SETTINGS_KEYS.DISABLE_REASONING_PARSING,
				label: 'Disable reasoning content parsing',
				type: SettingsFieldType.CHECKBOX
			},
			{
				key: SETTINGS_KEYS.EXCLUDE_REASONING_FROM_CONTEXT,
				label: 'Exclude reasoning from context',
				type: SettingsFieldType.CHECKBOX
			},
			{
				key: SETTINGS_KEYS.SHOW_RAW_OUTPUT_SWITCH,
				label: 'Enable raw output toggle',
				type: SettingsFieldType.CHECKBOX
			},
			{
				key: SETTINGS_KEYS.CUSTOM,
				label: 'Custom JSON',
				type: SettingsFieldType.TEXTAREA
			}
		]
	}
];
