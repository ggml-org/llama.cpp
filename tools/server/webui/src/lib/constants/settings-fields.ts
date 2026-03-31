/**
 * List of all numeric fields in settings configuration.
 * These fields will be converted from strings to numbers during save.
 */
export const NUMERIC_FIELDS = [
	'temperature',
	'top_k',
	'top_p',
	'min_p',
	'max_tokens',
	'pasteLongTextToFileLen',
	'dynatemp_range',
	'dynatemp_exponent',
	'typ_p',
	'xtc_probability',
	'xtc_threshold',
	'repeat_last_n',
	'repeat_penalty',
	'presence_penalty',
	'frequency_penalty',
	'dry_multiplier',
	'dry_base',
	'dry_allowed_length',
	'dry_penalty_last_n',
	'agenticMaxTurns',
	'agenticMaxToolPreviewLines'
] as const;

/**
 * Fields that must be positive integers (>= 1).
 * These will be clamped to minimum 1 and rounded during save.
 */
export const POSITIVE_INTEGER_FIELDS = ['agenticMaxTurns', 'agenticMaxToolPreviewLines'] as const;
