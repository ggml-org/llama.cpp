import { normalizeFloatingPoint } from '$lib/utils';
import { SETTINGS_KEYS } from '$lib/constants';
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
		key: SETTINGS_KEYS.TEMPERATURE,
		serverKey: SETTINGS_KEYS.TEMPERATURE,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.TOP_K,
		serverKey: SETTINGS_KEYS.TOP_K,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.TOP_P,
		serverKey: SETTINGS_KEYS.TOP_P,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.MIN_P,
		serverKey: SETTINGS_KEYS.MIN_P,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.DYNATEMP_RANGE,
		serverKey: SETTINGS_KEYS.DYNATEMP_RANGE,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.DYNATEMP_EXPONENT,
		serverKey: SETTINGS_KEYS.DYNATEMP_EXPONENT,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.XTC_PROBABILITY,
		serverKey: SETTINGS_KEYS.XTC_PROBABILITY,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.XTC_THRESHOLD,
		serverKey: SETTINGS_KEYS.XTC_THRESHOLD,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.TYP_P,
		serverKey: SETTINGS_KEYS.TYP_P,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.REPEAT_LAST_N,
		serverKey: SETTINGS_KEYS.REPEAT_LAST_N,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.REPEAT_PENALTY,
		serverKey: SETTINGS_KEYS.REPEAT_PENALTY,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.PRESENCE_PENALTY,
		serverKey: SETTINGS_KEYS.PRESENCE_PENALTY,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.FREQUENCY_PENALTY,
		serverKey: SETTINGS_KEYS.FREQUENCY_PENALTY,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.DRY_MULTIPLIER,
		serverKey: SETTINGS_KEYS.DRY_MULTIPLIER,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.DRY_BASE,
		serverKey: SETTINGS_KEYS.DRY_BASE,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.DRY_ALLOWED_LENGTH,
		serverKey: SETTINGS_KEYS.DRY_ALLOWED_LENGTH,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.DRY_PENALTY_LAST_N,
		serverKey: SETTINGS_KEYS.DRY_PENALTY_LAST_N,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.MAX_TOKENS,
		serverKey: SETTINGS_KEYS.MAX_TOKENS,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.SAMPLERS,
		serverKey: SETTINGS_KEYS.SAMPLERS,
		type: SyncableParameterType.STRING,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.BACKEND_SAMPLING,
		serverKey: SETTINGS_KEYS.BACKEND_SAMPLING,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.PASTE_LONG_TEXT_TO_FILE_LEN,
		serverKey: SETTINGS_KEYS.PASTE_LONG_TEXT_TO_FILE_LEN,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.PDF_AS_IMAGE,
		serverKey: SETTINGS_KEYS.PDF_AS_IMAGE,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.SHOW_THOUGHT_IN_PROGRESS,
		serverKey: SETTINGS_KEYS.SHOW_THOUGHT_IN_PROGRESS,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.KEEP_STATS_VISIBLE,
		serverKey: SETTINGS_KEYS.KEEP_STATS_VISIBLE,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.SHOW_MESSAGE_STATS,
		serverKey: SETTINGS_KEYS.SHOW_MESSAGE_STATS,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.ASK_FOR_TITLE_CONFIRMATION,
		serverKey: SETTINGS_KEYS.ASK_FOR_TITLE_CONFIRMATION,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.TITLE_GENERATION_USE_FIRST_LINE,
		serverKey: SETTINGS_KEYS.TITLE_GENERATION_USE_FIRST_LINE,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.DISABLE_AUTO_SCROLL,
		serverKey: SETTINGS_KEYS.DISABLE_AUTO_SCROLL,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.RENDER_USER_CONTENT_AS_MARKDOWN,
		serverKey: SETTINGS_KEYS.RENDER_USER_CONTENT_AS_MARKDOWN,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.AUTO_MIC_ON_EMPTY,
		serverKey: SETTINGS_KEYS.AUTO_MIC_ON_EMPTY,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.PY_INTERPRETER_ENABLED,
		serverKey: SETTINGS_KEYS.PY_INTERPRETER_ENABLED,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.ENABLE_CONTINUE_GENERATION,
		serverKey: SETTINGS_KEYS.ENABLE_CONTINUE_GENERATION,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.FULL_HEIGHT_CODE_BLOCKS,
		serverKey: SETTINGS_KEYS.FULL_HEIGHT_CODE_BLOCKS,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.SYSTEM_MESSAGE,
		serverKey: SETTINGS_KEYS.SYSTEM_MESSAGE,
		type: SyncableParameterType.STRING,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.SHOW_SYSTEM_MESSAGE,
		serverKey: SETTINGS_KEYS.SHOW_SYSTEM_MESSAGE,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.THEME,
		serverKey: SETTINGS_KEYS.THEME,
		type: SyncableParameterType.STRING,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.COPY_TEXT_ATTACHMENTS_AS_PLAIN_TEXT,
		serverKey: SETTINGS_KEYS.COPY_TEXT_ATTACHMENTS_AS_PLAIN_TEXT,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.SHOW_RAW_OUTPUT_SWITCH,
		serverKey: SETTINGS_KEYS.SHOW_RAW_OUTPUT_SWITCH,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.ALWAYS_SHOW_SIDEBAR_ON_DESKTOP,
		serverKey: SETTINGS_KEYS.ALWAYS_SHOW_SIDEBAR_ON_DESKTOP,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.SHOW_RAW_MODEL_NAMES,
		serverKey: SETTINGS_KEYS.SHOW_RAW_MODEL_NAMES,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.MCP_SERVERS,
		serverKey: SETTINGS_KEYS.MCP_SERVERS,
		type: SyncableParameterType.STRING,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.AGENTIC_MAX_TURNS,
		serverKey: SETTINGS_KEYS.AGENTIC_MAX_TURNS,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.AGENTIC_MAX_TOOL_PREVIEW_LINES,
		serverKey: SETTINGS_KEYS.AGENTIC_MAX_TOOL_PREVIEW_LINES,
		type: SyncableParameterType.NUMBER,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.SHOW_TOOL_CALL_IN_PROGRESS,
		serverKey: SETTINGS_KEYS.SHOW_TOOL_CALL_IN_PROGRESS,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.ALWAYS_SHOW_AGENTIC_TURNS,
		serverKey: SETTINGS_KEYS.ALWAYS_SHOW_AGENTIC_TURNS,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.EXCLUDE_REASONING_FROM_CONTEXT,
		serverKey: SETTINGS_KEYS.EXCLUDE_REASONING_FROM_CONTEXT,
		type: SyncableParameterType.BOOLEAN,
		canSync: true
	},
	{
		key: SETTINGS_KEYS.SEND_ON_ENTER,
		serverKey: SETTINGS_KEYS.SEND_ON_ENTER,
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
				extracted[SETTINGS_KEYS.SAMPLERS] = serverParams.samplers.join(';');
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
	 * Used by SettingsChatParameterSourceIndicator to display the correct badge
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
