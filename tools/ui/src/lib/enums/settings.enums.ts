/**
 * Parameter source - indicates whether a parameter uses default or custom value
 */
export enum ParameterSource {
	CUSTOM = 'custom',
	DEFAULT = 'default'
}

/**
 * Syncable parameter type - data types for parameters that can be synced with server
 */
export enum SyncableParameterType {
	BOOLEAN = 'boolean',
	NUMBER = 'number',
	STRING = 'string'
}

/**
 * Settings field type - defines the input type for settings fields
 */
/** How the models manager builds its sections below Loaded and Favorites. */
export enum ModelGroupingMode {
	/** One llama-compat block and one OAI-compat block, families inside each. */
	COMPAT = 'compat',
	/** One section per provider, families inside each. */
	PROVIDER = 'provider'
}

export enum SettingsFieldType {
	CHECKBOX = 'checkbox',
	INPUT = 'input',
	RADIO = 'radio',
	SELECT = 'select',
	TEXTAREA = 'textarea'
}
