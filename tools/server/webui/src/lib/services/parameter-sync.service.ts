import { normalizeFloatingPoint } from '$lib/utils';
import type { SyncableParameter, ParameterRecord, ParameterInfo, ParameterValue } from '$lib/types';
import { SyncableParameterType, ParameterSource } from '$lib/enums';

/**
 * Mapping of webui setting keys to server parameter keys.
 * Only parameters listed here can be synced from the server `/props` endpoint.
 * Each entry defines the webui key, corresponding server key, value type,
 * and whether sync is enabled.
 */
export const SYNCABLE_PARAMETERS: SyncableParameter[] = [
	{
		key: 'temperature',
		serverKey: 'temperature',
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{ key: 'top_k', serverKey: 'top_k', type: SyncableParameterType.NUMBER, canSync: true },
	{ key: 'top_p', serverKey: 'top_p', type: SyncableParameterType.NUMBER, canSync: true },
	{ key: 'min_p', serverKey: 'min_p', type: SyncableParameterType.NUMBER, canSync: true },
	{
		key: 'dynatemp_range',
		serverKey: 'dynatemp_range',
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: 'dynatemp_exponent',
		serverKey: 'dynatemp_exponent',
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: 'xtc_probability',
		serverKey: 'xtc_probability',
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: 'xtc_threshold',
		serverKey: 'xtc_threshold',
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{ key: 'typ_p', serverKey: 'typ_p', type: SyncableParameterType.NUMBER, canSync: true },
	{
		key: 'repeat_last_n',
		serverKey: 'repeat_last_n',
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: 'repeat_penalty',
		serverKey: 'repeat_penalty',
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: 'presence_penalty',
		serverKey: 'presence_penalty',
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: 'frequency_penalty',
		serverKey: 'frequency_penalty',
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: 'dry_multiplier',
		serverKey: 'dry_multiplier',
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{ key: 'dry_base', serverKey: 'dry_base', type: SyncableParameterType.NUMBER, canSync: true },
	{
		key: 'dry_allowed_length',
		serverKey: 'dry_allowed_length',
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: 'dry_penalty_last_n',
		serverKey: 'dry_penalty_last_n',
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{ key: 'max_tokens', serverKey: 'max_tokens', type: SyncableParameterType.NUMBER, canSync: true },
	{ key: 'samplers', serverKey: 'samplers', type: SyncableParameterType.STRING, canSync: true },
	{
		key: 'pasteLongTextToFileLen',
		serverKey: 'pasteLongTextToFileLen',
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: 'pdfAsImage',
		serverKey: 'pdfAsImage',
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: 'showThoughtInProgress',
		serverKey: 'showThoughtInProgress',
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: 'keepStatsVisible',
		serverKey: 'keepStatsVisible',
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: 'showMessageStats',
		serverKey: 'showMessageStats',
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: 'askForTitleConfirmation',
		serverKey: 'askForTitleConfirmation',
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: 'disableAutoScroll',
		serverKey: 'disableAutoScroll',
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: 'renderUserContentAsMarkdown',
		serverKey: 'renderUserContentAsMarkdown',
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: 'autoMicOnEmpty',
		serverKey: 'autoMicOnEmpty',
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: 'pyInterpreterEnabled',
		serverKey: 'pyInterpreterEnabled',
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: 'enableContinueGeneration',
		serverKey: 'enableContinueGeneration',
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: 'fullHeightCodeBlocks',
		serverKey: 'fullHeightCodeBlocks',
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	}
];

export class ParameterSyncService {
	/**
	 *
	 *
	 * Extraction
	 *
	 *
	 */

	/**
	 * Round floating-point numbers to avoid JavaScript precision issues.
	 * E.g., 0.1 + 0.2 = 0.30000000000000004 → 0.3
	 *
	 * @param value - Parameter value to normalize
	 * @returns Precision-normalized value
	 */
	private static roundFloatingPoint(value: ParameterValue): ParameterValue {
		return normalizeFloatingPoint(value) as ParameterValue;
	}

	/**
	 * Extract server default parameters that can be synced from `/props` response.
	 * Handles both generation settings parameters and webui-specific settings.
	 * Converts samplers array to semicolon-delimited string for UI display.
	 *
	 * @param serverParams - Raw generation settings from server `/props` endpoint
	 * @param webuiSettings - Optional webui-specific settings from server
	 * @returns Record of extracted parameter key-value pairs with normalized precision
	 */
	static extractServerDefaults(
		serverParams: ApiLlamaCppServerProps['default_generation_settings']['params'] | null,
		webuiSettings?: Record<string, string | number | boolean>
	): ParameterRecord {
		const extracted: ParameterRecord = {};

		if (serverParams) {
			for (const param of SYNCABLE_PARAMETERS) {
				if (param.canSync && param.serverKey in serverParams) {
					const value = (serverParams as unknown as Record<string, ParameterValue>)[
						param.serverKey
					];
					if (value !== undefined) {
						// Apply precision rounding to avoid JavaScript floating-point issues
						extracted[param.key] = this.roundFloatingPoint(value);
					}
				}
			}

			// Handle samplers array conversion to string
			if (serverParams.samplers && Array.isArray(serverParams.samplers)) {
				extracted.samplers = serverParams.samplers.join(';');
			}
		}

		if (webuiSettings) {
			for (const param of SYNCABLE_PARAMETERS) {
				if (param.canSync && param.serverKey in webuiSettings) {
					const value = webuiSettings[param.serverKey];
					if (value !== undefined) {
						extracted[param.key] = this.roundFloatingPoint(value);
					}
				}
			}
		}

		return extracted;
	}

	/**
	 *
	 *
	 * Merging
	 *
	 *
	 */

	/**
	 * Merge server defaults with current user settings.
	 * User overrides always take priority — only parameters not in `userOverrides`
	 * set will be updated from server defaults.
	 *
	 * @param currentSettings - Current parameter values in the settings store
	 * @param serverDefaults - Default values extracted from server props
	 * @param userOverrides - Set of parameter keys explicitly overridden by the user
	 * @returns Merged parameter record with user overrides preserved
	 */
	static mergeWithServerDefaults(
		currentSettings: ParameterRecord,
		serverDefaults: ParameterRecord,
		userOverrides: Set<string> = new Set()
	): ParameterRecord {
		const merged = { ...currentSettings };

		for (const [key, serverValue] of Object.entries(serverDefaults)) {
			// Only update if user hasn't explicitly overridden this parameter
			if (!userOverrides.has(key)) {
				merged[key] = this.roundFloatingPoint(serverValue);
			}
		}

		return merged;
	}

	/**
	 *
	 *
	 * Info
	 *
	 *
	 */

	/**
	 * Get parameter information including source and values.
	 * Used by ChatSettingsParameterSourceIndicator to display the correct badge
	 * (Custom vs Default) for each parameter in the settings UI.
	 *
	 * @param key - The parameter key to get info for
	 * @param currentValue - The current value of the parameter
	 * @param propsDefaults - Server default values from `/props`
	 * @param userOverrides - Set of parameter keys explicitly overridden by the user
	 * @returns Parameter info with source, server default, and user override values
	 */
	static getParameterInfo(
		key: string,
		currentValue: ParameterValue,
		propsDefaults: ParameterRecord,
		userOverrides: Set<string>
	): ParameterInfo {
		const hasPropsDefault = propsDefaults[key] !== undefined;
		const isUserOverride = userOverrides.has(key);

		// Simple logic: either using default (from props) or custom (user override)
		const source = isUserOverride ? ParameterSource.CUSTOM : ParameterSource.DEFAULT;

		return {
			value: currentValue,
			source,
			serverDefault: hasPropsDefault ? propsDefaults[key] : undefined, // Keep same field name for compatibility
			userOverride: isUserOverride ? currentValue : undefined
		};
	}

	/**
	 * Check if a parameter can be synced from server.
	 *
	 * @param key - The parameter key to check
	 * @returns True if the parameter is in the syncable parameters list
	 */
	static canSyncParameter(key: string): boolean {
		return SYNCABLE_PARAMETERS.some((param) => param.key === key && param.canSync);
	}

	/**
	 * Get all syncable parameter keys.
	 *
	 * @returns Array of parameter keys that can be synced from server
	 */
	static getSyncableParameterKeys(): string[] {
		return SYNCABLE_PARAMETERS.filter((param) => param.canSync).map((param) => param.key);
	}

	/**
	 * Validate a server parameter value against its expected type.
	 *
	 * @param key - The parameter key to validate
	 * @param value - The value to validate
	 * @returns True if value matches the expected type for this parameter
	 */
	static validateServerParameter(key: string, value: ParameterValue): boolean {
		const param = SYNCABLE_PARAMETERS.find((p) => p.key === key);
		if (!param) return false;

		switch (param.type) {
			case SyncableParameterType.NUMBER:
				return typeof value === 'number' && !isNaN(value);
			case SyncableParameterType.STRING:
				return typeof value === 'string';
			case SyncableParameterType.BOOLEAN:
				return typeof value === 'boolean';
			default:
				return false;
		}
	}

	/**
	 *
	 *
	 * Diff
	 *
	 *
	 */

	/**
	 * Create a diff between current settings and server defaults.
	 * Shows which parameters differ from server values, useful for debugging
	 * and for the "Reset to defaults" functionality.
	 *
	 * @param currentSettings - Current parameter values in the settings store
	 * @param serverDefaults - Default values extracted from server props
	 * @returns Record of parameter diffs with current value, server value, and whether they differ
	 */
	static createParameterDiff(
		currentSettings: ParameterRecord,
		serverDefaults: ParameterRecord
	): Record<string, { current: ParameterValue; server: ParameterValue; differs: boolean }> {
		const diff: Record<
			string,
			{ current: ParameterValue; server: ParameterValue; differs: boolean }
		> = {};

		for (const key of this.getSyncableParameterKeys()) {
			const currentValue = currentSettings[key];
			const serverValue = serverDefaults[key];

			if (serverValue !== undefined) {
				diff[key] = {
					current: currentValue,
					server: serverValue,
					differs: currentValue !== serverValue
				};
			}
		}

		return diff;
	}
}
