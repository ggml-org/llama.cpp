import { getContext, setContext } from 'svelte';
import { CONTEXT_KEY_CHAT_SETTINGS_CONFIG } from '$lib/constants';

export interface ChatSettingsConfigContext {
	readonly localConfig: SettingsConfigType;
	handleConfigChange: (key: string, value: string | boolean) => void;
	handleThemeChange: (theme: string) => void;
}

const CHAT_SETTINGS_CONFIG_KEY = Symbol.for(CONTEXT_KEY_CHAT_SETTINGS_CONFIG);

export function setChatSettingsConfigContext(
	ctx: ChatSettingsConfigContext
): ChatSettingsConfigContext {
	return setContext(CHAT_SETTINGS_CONFIG_KEY, ctx);
}

export function getChatSettingsConfigContext(): ChatSettingsConfigContext {
	return getContext(CHAT_SETTINGS_CONFIG_KEY);
}
