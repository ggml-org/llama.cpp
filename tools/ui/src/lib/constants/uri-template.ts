/**
 * URI Template constants for RFC 6570 template processing.
 */

/** URI scheme separator */
export const URI_SCHEME_SEPARATOR = '://';

/** Regex to match template expressions like {var}, {+var}, {#var}, {/var} */
export const TEMPLATE_EXPRESSION_REGEX = /\{([+#./;?&]?)([^}]+)\}/g;

/** RFC 6570 URI template operators */
export const URI_TEMPLATE_OPERATORS = {
	/** Simple string expansion (default) */
	SIMPLE: '',
	/** Reserved expansion */
	RESERVED: '+',
	/** Fragment expansion */
	FRAGMENT: '#',
	/** Path segment expansion */
	PATH_SEGMENT: '/',
	/** Label expansion */
	LABEL: '.',
	/** Path-style parameters */
	PATH_PARAM: ';',
	/** Form-style query */
	FORM_QUERY: '?',
	/** Form-style query continuation */
	FORM_CONTINUATION: '&'
} as const;

/** URI template separators used in expansion */
export const URI_TEMPLATE_SEPARATORS = {
	/** Comma separator for list expansion */
	COMMA: ',',
	/** Slash separator for path segments */
	SLASH: '/',
	/** Period separator for label expansion */
	PERIOD: '.',
	/** Semicolon separator for path parameters */
	SEMICOLON: ';',
	/** Question mark prefix for query string */
	QUERY_PREFIX: '?',
	/** Ampersand prefix for query continuation */
	QUERY_CONTINUATION: '&'
} as const;

/** Maximum number of leading slashes to strip during URI normalization */
export const MAX_LEADING_SLASHES_TO_STRIP = 3;

/** Regex to strip explode modifier (*) from variable names */
export const VARIABLE_EXPLODE_MODIFIER_REGEX = /[*]$/;

/** Regex to strip prefix modifier (:N) from variable names */
export const VARIABLE_PREFIX_MODIFIER_REGEX = /:[\d]+$/;

/** Regex to strip one or more leading slashes */
export const LEADING_SLASHES_REGEX = /^\/+/;
