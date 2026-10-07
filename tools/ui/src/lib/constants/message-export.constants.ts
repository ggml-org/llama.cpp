// Conversation exporter / filename constants

export const EXPORT_CONV = {
	// Producer marker carried by the session record of a JSONL export
	HARNESS: 'llama.app',
	// Length of the trimmed conversation ID in the filename
	ID_TRIM_LENGTH: 8,
	// Replacements to the ISO date for use in the export filename
	ISO_DATE_TIME_SEPARATOR: 'T',

	ISO_DATE_TIME_SEPARATOR_REPLACEMENT: '_',

	ISO_TIME_SEPARATOR: ':',
	ISO_TIME_SEPARATOR_REPLACEMENT: '-',
	// Characters to keep in the ISO timestamp. 19 keeps 2026-01-01T00:00:00
	ISO_TIMESTAMP_SLICE: 19,

	MULTIPLE_UNDERSCORE_REGEX: /_+/g,
	// Maximum length of the sanitized conversation name snippet
	NAME_SUFFIX_MAX_LENGTH: 20,
	// Replacements for making the conversation title filename-friendly
	NON_ALPHANUMERIC_REGEX: /[^a-z0-9]/gi,
	NONALNUM_REPLACEMENT: '_'
} as const;
