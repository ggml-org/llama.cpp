import { parseMcpServerSettings } from './mcp';
import { canonicalizeServerUrl } from './url';
import {
	MCP_SERVER_ID_PREFIX,
	RECOMMENDED_MCP_SERVERS,
	SETTING_CONFIG_DEFAULT,
	SETTING_PRESET_DEFAULTS,
	SETTINGS_KEYS,
	UI_PRESETS
} from '$lib/constants';
import { UiPreset } from '$lib/enums';
import type { SettingsConfigValue } from '$lib/types';

export function isUiPreset(value: unknown): value is UiPreset {
	return Object.values(UiPreset).includes(value as UiPreset);
}

/**
 * Factory defaults with the preset-aware settings resolved for the given preset.
 * No preset (not picked yet) gives the minimal defaults.
 */
export function getSettingsDefaults(preset: unknown): Record<string, SettingsConfigValue> {
	const defaults = { ...SETTING_CONFIG_DEFAULT };

	if (!isUiPreset(preset)) return defaults;

	for (const [key, presetValues] of Object.entries(SETTING_PRESET_DEFAULTS)) {
		if (preset in presetValues) {
			defaults[key] = presetValues[preset];
		}
	}

	return defaults;
}

/**
 * Config changes that apply a preset: the preset key, the preset-aware settings and its MCP servers.
 * With onlyNonMinimal, settings at their minimal value are not written, so the preset only turns features on.
 * MCP servers are only added, never removed; a server already configured (same URL) is skipped.
 */
export function getUiPresetConfig(
	preset: UiPreset,
	config: SettingsConfigType,
	onlyNonMinimal = false
): Record<string, SettingsConfigValue> {
	const defaults = getSettingsDefaults(preset);
	const updates: Record<string, SettingsConfigValue> = { [SETTINGS_KEYS.UI_PRESET]: preset };

	for (const key of Object.keys(SETTING_PRESET_DEFAULTS)) {
		if (onlyNonMinimal && defaults[key] === SETTING_CONFIG_DEFAULT[key]) continue;

		updates[key] = defaults[key];
	}

	const servers = parseMcpServerSettings(config[SETTINGS_KEYS.MCP_SERVERS]);
	const configuredUrls = new Set(servers.map((s) => canonicalizeServerUrl(s.url)));
	const newServers = RECOMMENDED_MCP_SERVERS.filter(
		(rec) =>
			UI_PRESETS[preset].mcpServerIds.includes(rec.id) &&
			!configuredUrls.has(canonicalizeServerUrl(rec.url))
	).map((rec) => ({
		enabled: true,
		id: `${MCP_SERVER_ID_PREFIX}-${rec.id}`,
		url: rec.url,
		useProxy: false
	}));

	if (newServers.length > 0) {
		updates[SETTINGS_KEYS.MCP_SERVERS] = JSON.stringify([...servers, ...newServers]);
	}

	return updates;
}
