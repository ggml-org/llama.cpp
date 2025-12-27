/**
 * ParameterSyncService - Handles synchronization between server defaults and user settings
 *
 * This service manages the complex logic of merging server-provided default parameters
 * with user-configured overrides, ensuring the UI reflects the actual server state
 * while preserving user customizations.
 *
 * **Key Responsibilities:**
 * - Extract syncable parameters from server props
 * - Merge server defaults with user overrides
 * - Track parameter sources (server, user, default)
 * - Provide sync utilities for settings store integration
 */

import { normalizeFloatingPoint } from '$lib/utils';

export type ParameterSource = 'default' | 'custom';
export type ParameterValue = string | number | boolean;
export type ParameterRecord = Record<string, ParameterValue>;

export interface ParameterInfo {
	value: string | number | boolean;
	source: ParameterSource;
	serverDefault?: string | number | boolean;
	userOverride?: string | number | boolean;
}

export interface SyncableParameter {
	key: string;
	serverKey: string;
	type: 'number' | 'string' | 'boolean';
}

/**
 * Mapping of webui setting keys to server parameter keys
 * Only parameters that should be synced from server are included
 */
export const SYNCABLE_PARAMETERS: SyncableParameter[] = [
	{ key: 'apiKey', serverKey: 'apiKey', type: 'string' },
	{ key: 'systemMessage', serverKey: 'systemMessage', type: 'string' },
	{ key: 'showSystemMessage', serverKey: 'showSystemMessage', type: 'boolean' },
	{ key: 'theme', serverKey: 'theme', type: 'string' },
	{ key: 'temperature', serverKey: 'temperature', type: 'number' },
	{ key: 'top_k', serverKey: 'top_k', type: 'number' },
	{ key: 'top_p', serverKey: 'top_p', type: 'number' },
	{ key: 'min_p', serverKey: 'min_p', type: 'number' },
	{ key: 'dynatemp_range', serverKey: 'dynatemp_range', type: 'number' },
	{ key: 'dynatemp_exponent', serverKey: 'dynatemp_exponent', type: 'number' },
	{ key: 'xtc_probability', serverKey: 'xtc_probability', type: 'number' },
	{ key: 'xtc_threshold', serverKey: 'xtc_threshold', type: 'number' },
	{ key: 'typ_p', serverKey: 'typ_p', type: 'number' },
	{ key: 'repeat_last_n', serverKey: 'repeat_last_n', type: 'number' },
	{ key: 'repeat_penalty', serverKey: 'repeat_penalty', type: 'number' },
	{ key: 'presence_penalty', serverKey: 'presence_penalty', type: 'number' },
	{ key: 'frequency_penalty', serverKey: 'frequency_penalty', type: 'number' },
	{ key: 'dry_multiplier', serverKey: 'dry_multiplier', type: 'number' },
	{ key: 'dry_base', serverKey: 'dry_base', type: 'number' },
	{ key: 'dry_allowed_length', serverKey: 'dry_allowed_length', type: 'number' },
	{ key: 'dry_penalty_last_n', serverKey: 'dry_penalty_last_n', type: 'number' },
	{ key: 'max_tokens', serverKey: 'max_tokens', type: 'number' },
	{ key: 'samplers', serverKey: 'samplers', type: 'string' },
	{
		key: 'pasteLongTextToFileLen',
		serverKey: 'pasteLongTextToFileLen',
		type: 'number'
	},
	{
		key: 'copyTextAttachmentsAsPlainText',
		serverKey: 'copyTextAttachmentsAsPlainText',
		type: 'boolean'
	},
	{ key: 'pdfAsImage', serverKey: 'pdfAsImage', type: 'boolean' },
	{
		key: 'showThoughtInProgress',
		serverKey: 'showThoughtInProgress',
		type: 'boolean'
	},
	{ key: 'showToolCalls', serverKey: 'showToolCalls', type: 'boolean' },
	{
		key: 'disableReasoningFormat',
		serverKey: 'disableReasoningFormat',
		type: 'boolean'
	},
	{
		key: 'alwaysShowSidebarOnDesktop',
		serverKey: 'alwaysShowSidebarOnDesktop',
		type: 'boolean'
	},
	{
		key: 'autoShowSidebarOnNewChat',
		serverKey: 'autoShowSidebarOnNewChat',
		type: 'boolean'
	},
	{ key: 'keepStatsVisible', serverKey: 'keepStatsVisible', type: 'boolean' },
	{ key: 'showMessageStats', serverKey: 'showMessageStats', type: 'boolean' },
	{
		key: 'askForTitleConfirmation',
		serverKey: 'askForTitleConfirmation',
		type: 'boolean'
	},
	{ key: 'disableAutoScroll', serverKey: 'disableAutoScroll', type: 'boolean' },
	{
		key: 'renderUserContentAsMarkdown',
		serverKey: 'renderUserContentAsMarkdown',
		type: 'boolean'
	},
	{ key: 'autoMicOnEmpty', serverKey: 'autoMicOnEmpty', type: 'boolean' },
	{ key: 'custom', serverKey: 'custom', type: 'string' },
	{
		key: 'pyInterpreterEnabled',
		serverKey: 'pyInterpreterEnabled',
		type: 'boolean'
	},
	{
		key: 'enableContinueGeneration',
		serverKey: 'enableContinueGeneration',
		type: 'boolean'
	}
];

export class ParameterSyncService {
	// ─────────────────────────────────────────────────────────────────────────────
	// Extraction
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Round floating-point numbers to avoid JavaScript precision issues
	 */
	private static roundFloatingPoint(value: ParameterValue): ParameterValue {
		return normalizeFloatingPoint(value) as ParameterValue;
	}

	/**
	 * Extract server default parameters that can be synced
	 */
	static extractServerDefaults(
		serverParams: ApiLlamaCppServerProps['default_generation_settings']['params'] | null,
		webuiSettings?: Record<string, string | number | boolean>
	): ParameterRecord {
		const extracted: ParameterRecord = {};

		if (serverParams) {
			for (const param of SYNCABLE_PARAMETERS) {
				if (param.serverKey in serverParams) {
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
				if (param.serverKey in webuiSettings) {
					const value = webuiSettings[param.serverKey];
					if (value !== undefined) {
						extracted[param.key] = this.roundFloatingPoint(value);
					}
				}
			}
		}

		return extracted;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Merging
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Merge server defaults with current user settings
	 * Returns updated settings that respect user overrides while using server defaults
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

	// ─────────────────────────────────────────────────────────────────────────────
	// Info
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Get parameter information including source and values
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
		const source: ParameterSource = isUserOverride ? 'custom' : 'default';

		return {
			value: currentValue,
			source,
			serverDefault: hasPropsDefault ? propsDefaults[key] : undefined, // Keep same field name for compatibility
			userOverride: isUserOverride ? currentValue : undefined
		};
	}

	/**
	 * Check if a parameter can be synced from server
	 */
	static canSyncParameter(key: string): boolean {
		return SYNCABLE_PARAMETERS.some((param) => param.key === key);
	}

	/**
	 * Get all syncable parameter keys
	 */
	static getSyncableParameterKeys(): string[] {
		return SYNCABLE_PARAMETERS.map((param) => param.key);
	}

	/**
	 * Validate server parameter value
	 */
	static validateServerParameter(key: string, value: ParameterValue): boolean {
		const param = SYNCABLE_PARAMETERS.find((p) => p.key === key);
		if (!param) return false;

		switch (param.type) {
			case 'number':
				return typeof value === 'number' && !isNaN(value);
			case 'string':
				return typeof value === 'string';
			case 'boolean':
				return typeof value === 'boolean';
			default:
				return false;
		}
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Diff
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Create a diff between current settings and server defaults
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
