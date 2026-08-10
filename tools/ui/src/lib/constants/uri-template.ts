/**
 * URI Template constants for RFC 6570 template processing.
 */

/** URI scheme separator */
export const URI_SCHEME_SEPARATOR = '://';

/** Regex to match template expressions like {var}, {+var}, {#var}, {/var} */
export const TEMPLATE_EXPRESSION_REGEX = /\{([+#./;?&]?)([^}]+)\}/g;

/** RFC 6570 URI template operators */
export const URI_TEMPLATE_OPERATORS = {
	/** Form-style query continuation */
	FORM_CONTINUATION: '&',
	/** Form-style query */
	FORM_QUERY: '?',
	/** Fragment expansion */
	FRAGMENT: '#',
	/** Label expansion */
	LABEL: '.',
	/** Path-style parameters */
	PATH_PARAM: ';',
	/** Path segment expansion */
	PATH_SEGMENT: '/',
	/** Reserved expansion */
	RESERVED: '+',
	/** Simple string expansion (default) */
	SIMPLE: ''
} as const;

/** URI template separators used in expansion */
export const URI_TEMPLATE_SEPARATORS = {
	/** Comma separator for list expansion */
	COMMA: ',',
	/** Period separator for label expansion */
	PERIOD: '.',
	/** Ampersand prefix for query continuation */
	QUERY_CONTINUATION: '&',
	/** Question mark prefix for query string */
	QUERY_PREFIX: '?',
	/** Semicolon separator for path parameters */
	SEMICOLON: ';',
	/** Slash separator for path segments */
	SLASH: '/'
} as const;

/** Maximum number of leading slashes to strip during URI normalization */
export const MAX_LEADING_SLASHES_TO_STRIP = 3;

/** Regex to strip explode modifier (*) from variable names */
export const VARIABLE_EXPLODE_MODIFIER_REGEX = /[*]$/;

/** Regex to strip prefix modifier (:N) from variable names */
export const VARIABLE_PREFIX_MODIFIER_REGEX = /:[\d]+$/;

/** Regex to strip one or more leading slashes */
export const LEADING_SLASHES_REGEX = /^\/+/;

/** Regex to match base64-encoded image URIs (format: "data:image/[media type];base64,[data]")*/
export const BASE64_IMAGE_URI_REGEX = /^data:(image\/[a-z0-9.\-+]+);base64/;
