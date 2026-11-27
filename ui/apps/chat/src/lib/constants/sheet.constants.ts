/**
 * The phone sheets (settings, add to chat, the model picker): their sizes and their text. They are
 * built from @ggml-org/llama-ui/app/sheets.
 */

import { ColorMode } from '#lib/enums/ui.enums.ts';

/** A sheet as tall as the page under the status bar's corner controls. */
export const SHEET_FULL_HEIGHT_CLASS =
	'data-[vaul-drawer-direction=bottom]:h-[calc(100dvh-var(--corner-control-top))] data-[vaul-drawer-direction=bottom]:max-h-none';

/** A sheet as tall as its content, up to the full height. */
export const SHEET_AUTO_HEIGHT_CLASS =
	'data-[vaul-drawer-direction=bottom]:h-auto data-[vaul-drawer-direction=bottom]:max-h-[calc(100dvh-var(--corner-control-top))]';

/** A short sheet: as tall as its content, leaving the page visible above it. */
export const SHEET_SHORT_HEIGHT_CLASS = 'data-[vaul-drawer-direction=bottom]:max-h-[85dvh]';

/** How many models a phone's picker opens with, before its full list. */
export const MODELS_SHEET_QUICK_COUNT = 6;

/** Keys of the model picker's groups that are not a model family. */
export const MODELS_SHEET_GROUP_KEYS = {
	FAMILY_PREFIX: 'family-',
	FAVORITES: 'favorites',
	LOADED: 'loaded',
	LOCAL: 'local'
} as const;

export const MODELS_SHEET_TEXT = {
	ALL_MODELS: 'All models',
	CANCEL_LOADING: 'Cancel loading',
	DESCRIPTION: 'Pick the model for this chat.',
	DICTATION: 'Dictation',
	EMPTY: 'No models found.',
	FAVORITES: 'Favorites',
	IN_CHAT: 'In chat',
	LOADED: 'Loaded',
	LOCAL_MODELS: 'Local models',
	MODEL_ACTIONS: 'Model actions',
	MORE_DETAILS: 'More details',
	NOT_AVAILABLE: 'Not available',
	SEARCH_PLACEHOLDER: 'Search models',
	SELECT_MODEL: 'Select model',
	UNLOAD: 'Unload'
} as const;

export const SETTINGS_SHEET_TEXT = {
	APP: 'App',
	APPEARANCE: 'Appearance',
	CANCEL: 'Cancel',
	CUSTOM_JSON_INVALID: 'Custom JSON is not valid JSON, so it was not saved',
	DESCRIPTION: 'How the app looks, chats and talks to the server.',
	MCP_SERVERS: 'MCP servers',
	NEEDS_NUMBER: (label: string) => `${label} needs a number`,
	RELOAD_APP: 'Reload app',
	RESET: 'Reset settings to default',
	RESET_CONFIRM: 'Reset to Default',
	RESET_DESCRIPTION:
		"All settings go back to their default values: the server's /props values where it has them. Your conversations are kept.",
	RESET_DONE: 'Settings reset to their defaults',
	RESET_FIELD: (label: string) => `Reset ${label} to default`,
	RESET_TITLE: 'Reset Settings to Default',
	SAVED_NOTE: 'Settings are saved in this browser.',
	SHOW_SYSTEM_MESSAGE: 'Show system message in conversations',
	TITLE: 'Settings',
	TOOL: 'Tool',
	TOOLS: 'Tools'
} as const;

/** The first page's theme tiles, in the order they are shown. */
export const SETTINGS_SHEET_THEMES = [
	{ label: 'Light', value: ColorMode.LIGHT },
	{ label: 'Dark', value: ColorMode.DARK },
	{ label: 'System', value: ColorMode.SYSTEM }
] as const;

/** Labels of the phone sidebar's pinned keys. */
export const SIDEBAR_KEYS_TEXT = {
	NEW_CHAT: 'New chat',
	SETTINGS: 'Settings'
} as const;
