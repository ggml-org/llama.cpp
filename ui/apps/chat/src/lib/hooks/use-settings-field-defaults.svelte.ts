import { SETTINGS_KEYS } from '#lib/constants/index.ts';
import { SettingsFieldType } from '#lib/enums/settings.enums.ts';
import { modelsStore, serverStore } from '#lib/stores/index.ts';
import { normalizeFloatingPoint } from '#lib/utils/precision.ts';

/**
 * What the settings fields share on a desktop and a phone: the server's defaults for the loaded
 * model, whether a value differs from them, and which fields another setting locks.
 */
export function useSettingsFieldDefaults(getConfig: () => SettingsConfigType) {
	const serverDefaults = $derived.by(() => {
		void modelsStore.props.cacheVersion;

		if (serverStore.isRouterMode) {
			const modelName = modelsStore.selectedModelName;

			if (modelName) {
				const modelProps = modelsStore.props.getModelProps(modelName);

				return (modelProps?.default_generation_settings?.params ?? {}) as Record<string, unknown>;
			}
		}

		return (serverStore.defaultParams ?? {}) as Record<string, unknown>;
	});
	// discover needs the hub metadata reads, and opens from the Models page, so both are read-only while it is on
	const discoverOn = $derived(Boolean(getConfig()[SETTINGS_KEYS.ENABLE_DISCOVER_MODELS]));

	return {
		/** A value set apart from the server's default for the loaded model. */
		isCustom(field: Pick<SettingsFieldConfig, 'key' | 'type'>): boolean {
			const serverDefault = serverDefaults[field.key];
			const value = getConfig()[field.key];

			if (serverDefault == null || value === '' || value === undefined) return false;

			if (field.type === SettingsFieldType.SELECT) return value !== serverDefault;

			const numeric = parseFloat(String(value));

			return (
				(isNaN(numeric) ? String(value) : normalizeFloatingPoint(numeric)) !==
				normalizeFloatingPoint(serverDefault)
			);
		},
		isLocked(key: string): boolean {
			return (
				discoverOn &&
				(key === SETTINGS_KEYS.USE_HUGGING_FACE_HUB || key === SETTINGS_KEYS.SHOW_MODELS_MANAGER)
			);
		},
		get serverDefaults() {
			return serverDefaults;
		}
	};
}
