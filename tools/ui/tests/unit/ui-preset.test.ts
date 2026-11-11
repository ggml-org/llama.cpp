import { SETTING_CONFIG_DEFAULT, SETTINGS_KEYS } from '$lib/constants';
import { UiPreset } from '$lib/enums';
import { getSettingsDefaults, getUiPresetConfig } from '$lib/utils';
import { describe, expect, it } from 'vitest';

const EXA_URL = 'https://mcp.exa.ai/mcp';

function config(overrides: Record<string, unknown> = {}): SettingsConfigType {
	return { ...SETTING_CONFIG_DEFAULT, ...overrides } as SettingsConfigType;
}

function mcpUrls(value: unknown): string[] {
	return (JSON.parse(value as string) as Array<{ url: string }>).map((s) => s.url);
}

describe('getSettingsDefaults', () => {
	it('keeps the factory defaults for minimal or no preset', () => {
		expect(getSettingsDefaults(undefined)).toEqual(SETTING_CONFIG_DEFAULT);
		expect(getSettingsDefaults(UiPreset.MINIMAL)).toEqual(SETTING_CONFIG_DEFAULT);
		expect(getSettingsDefaults('unknown')).toEqual(SETTING_CONFIG_DEFAULT);
	});

	it('resolves comfortable without the full-only features', () => {
		const defaults = getSettingsDefaults(UiPreset.COMFORTABLE);

		expect(defaults[SETTINGS_KEYS.JS_SANDBOX_ENABLED]).toBe(true);
		expect(defaults[SETTINGS_KEYS.SYMBOLIC_MATH_ENABLED]).toBe(false);
		expect(defaults[SETTINGS_KEYS.TITLE_GENERATION_USE_LLM]).toBe(true);
		expect(defaults[SETTINGS_KEYS.TITLE_GENERATION_USE_FIRST_LINE]).toBe(false);
		expect(defaults[SETTINGS_KEYS.ENABLE_DISCOVER_MODELS]).toBe(true);
		expect(defaults[SETTINGS_KEYS.USE_HUGGING_FACE_HUB]).toBe(true);
	});

	it('resolves full with the extra features', () => {
		const defaults = getSettingsDefaults(UiPreset.FULL);

		expect(defaults[SETTINGS_KEYS.SYMBOLIC_MATH_ENABLED]).toBe(true);
		expect(defaults[SETTINGS_KEYS.ENABLE_CONTINUE_GENERATION]).toBe(true);
	});
});

describe('getUiPresetConfig', () => {
	it('stores the preset and adds its MCP servers', () => {
		const updates = getUiPresetConfig(UiPreset.COMFORTABLE, config());

		expect(updates[SETTINGS_KEYS.UI_PRESET]).toBe(UiPreset.COMFORTABLE);
		expect(mcpUrls(updates[SETTINGS_KEYS.MCP_SERVERS])).toEqual([EXA_URL]);
	});

	it('does not add a server that is already configured', () => {
		const existing = JSON.stringify([{ enabled: false, id: 'mine', url: `${EXA_URL}/` }]);
		const updates = getUiPresetConfig(
			UiPreset.COMFORTABLE,
			config({ [SETTINGS_KEYS.MCP_SERVERS]: existing })
		);

		expect(updates[SETTINGS_KEYS.MCP_SERVERS]).toBeUndefined();
	});

	it('turns features off when switching down to minimal', () => {
		const updates = getUiPresetConfig(
			UiPreset.MINIMAL,
			config({ [SETTINGS_KEYS.JS_SANDBOX_ENABLED]: true })
		);

		expect(updates[SETTINGS_KEYS.JS_SANDBOX_ENABLED]).toBe(false);
		expect(updates[SETTINGS_KEYS.MCP_SERVERS]).toBeUndefined();
	});

	it('only turns features on with onlyNonMinimal', () => {
		const minimal = getUiPresetConfig(
			UiPreset.MINIMAL,
			config({ [SETTINGS_KEYS.JS_SANDBOX_ENABLED]: true }),
			true
		);

		expect(minimal).toEqual({ [SETTINGS_KEYS.UI_PRESET]: UiPreset.MINIMAL });

		const comfortable = getUiPresetConfig(UiPreset.COMFORTABLE, config(), true);

		expect(comfortable[SETTINGS_KEYS.JS_SANDBOX_ENABLED]).toBe(true);
		expect(SETTINGS_KEYS.SYMBOLIC_MATH_ENABLED in comfortable).toBe(false);
	});
});
