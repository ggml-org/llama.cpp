/**
 * String patterns for detecting content kind from MIME types and URIs.
 * Used with startsWith/includes checks, not as discriminated values.
 */

export const MIME_TYPE_PREFIXES = {
	IMAGE: 'image/',
	TEXT: 'text'
} as const;

export const MIME_TYPE_SUBSTRINGS = {
	JAVASCRIPT: 'javascript',
	JSON: 'json',
	TYPESCRIPT: 'typescript'
} as const;

export const URI_PATTERNS = {
	DATABASE_KEYWORD: 'database',
	DATABASE_SCHEME: 'db://'
} as const;
