import { getContext, setContext } from 'svelte';
import type { SettingsSectionTitle } from '$lib/constants';
import { CONTEXT_KEY_CHAT_SETTINGS_DIALOG } from '$lib/constants';

export interface ChatSettingsDialogContext {
	open: (initialSection?: SettingsSectionTitle) => void;
}

const CHAT_SETTINGS_DIALOG_KEY = Symbol.for(CONTEXT_KEY_CHAT_SETTINGS_DIALOG);

export function setChatSettingsDialogContext(
	ctx: ChatSettingsDialogContext
): ChatSettingsDialogContext {
	return setContext(CHAT_SETTINGS_DIALOG_KEY, ctx);
}

export function getChatSettingsDialogContext(): ChatSettingsDialogContext {
	return getContext(CHAT_SETTINGS_DIALOG_KEY);
}
