// Conversation filename constants

// Length of the trimmed conversation ID in the filename
export const EXPORT_CONV_ID_TRIM_LENGTH = 8;
// Maximum length of the sanitized conversation name snippet
export const EXPORT_CONV_NAME_SUFFIX_MAX_LENGTH = 20;
// Characters to keep in the ISO timestamp. 19 keeps 2026-01-01T00:00:00
export const ISO_TIMESTAMP_SLICE_LENGTH = 19;

export const EXPORT_CONV_NONALNUM_REPLACEMENT = '_';

export const ISO_DATE_TIME_SEPARATOR = 'T';
export const ISO_DATE_TIME_SEPARATOR_REPLACEMENT = '_';

export const ISO_TIME_SEPARATOR = ':';
export const ISO_TIME_SEPARATOR_REPLACEMENT = '-';

export const NON_ALPHANUMERIC_REGEX = /[^a-z0-9]/gi;
export const MULTIPLE_UNDERSCORE_REGEX = /_+/g;
