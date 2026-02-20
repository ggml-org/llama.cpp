/**
 * Settings section titles constants for ChatSettings component.
 */
export const SETTINGS_SECTION_TITLES = {
	GENERAL: 'General',
	DISPLAY: 'Display',
	SAMPLING: 'Sampling',
	PENALTIES: 'Penalties',
	IMPORT_EXPORT: 'Import/Export',
	DEVELOPER: 'Developer'
} as const;

export type SettingsSectionTitle =
	(typeof SETTINGS_SECTION_TITLES)[keyof typeof SETTINGS_SECTION_TITLES];
