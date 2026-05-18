import {
	TEMPLATE_EXPRESSION_REGEX,
	URI_SCHEME_SEPARATOR,
	URI_TEMPLATE_OPERATORS,
	URI_TEMPLATE_SEPARATORS,
	VARIABLE_EXPLODE_MODIFIER_REGEX,
	VARIABLE_PREFIX_MODIFIER_REGEX,
	LEADING_SLASHES_REGEX
} from '../constants';

/**
 * Normalize a resource URI for comparison.
 *
 * URI template expansion (especially with path operators like {/var})
 * can produce URIs that differ from listed resource URIs in slash placement.
 * For example, the template `svelte://{/slug*}.md` with slug="svelte/$effect"
 * expands to `svelte:///svelte/$effect.md`, while the listed resource URI is
 * `svelte://svelte/$effect.md`.
 *
 * This function strips extra leading slashes after the scheme to normalize
 * both forms to the same string for comparison purposes.
 *
 * @param uri - The URI to normalize
 * @returns Normalized URI string
 */
export function normalizeResourceUri(uri: string): string {
	const schemeEnd = uri.indexOf(URI_SCHEME_SEPARATOR);
	if (schemeEnd === -1) return uri;

	const scheme = uri.substring(0, schemeEnd);
	const rest = uri
		.substring(schemeEnd + URI_SCHEME_SEPARATOR.length)
		.replace(LEADING_SLASHES_REGEX, '');

	return `${scheme}${URI_SCHEME_SEPARATOR}${rest}`;
}

/**
 * A parsed variable from a URI template expression.
 */
export interface UriTemplateVariable {
	/** Variable name */
	name: string;
	/** Operator prefix (+, #, /, etc.) or empty string */
	operator: string;
}

/**
 * Extract all variable names from a URI template string.
 *
 * @param template - URI template string (RFC 6570)
 * @returns Array of unique variable descriptors
 *
 * @example
 * ```ts
 * extractTemplateVariables("file:///{path}")
 * // => [{ name: "path", operator: "" }]
 *
 * extractTemplateVariables("db://{schema}/{table}")
 * // => [{ name: "schema", operator: "" }, { name: "table", operator: "" }]
 * ```
 */
export function extractTemplateVariables(template: string): UriTemplateVariable[] {
	const variables: UriTemplateVariable[] = [];
	const seen = new Set<string>();

	let match;
	TEMPLATE_EXPRESSION_REGEX.lastIndex = 0;

	while ((match = TEMPLATE_EXPRESSION_REGEX.exec(template)) !== null) {
		const operator = match[1] || '';
		const varList = match[2];

		// RFC 6570 allows comma-separated variable lists: {x,y,z}
		for (const varSpec of varList.split(',')) {
			// Strip explode modifier (*) and prefix modifier (:N)
			const name = varSpec
				.replace(VARIABLE_EXPLODE_MODIFIER_REGEX, '')
				.replace(VARIABLE_PREFIX_MODIFIER_REGEX, '')
				.trim();

			if (name && !seen.has(name)) {
				seen.add(name);
				variables.push({ name, operator });
			}
		}
	}

	return variables;
}

/**
 * Expand a URI template with the given variable values.
 * Implements a simplified RFC 6570 Level 2 expansion.
 *
 * @param template - URI template string
 * @param values - Map of variable name to value
 * @returns Expanded URI string
 *
 * @example
 * ```ts
 * expandTemplate("file:///{path}", { path: "src/main.rs" })
 * // => "file:///src/main.rs"
 * ```
 */
export function expandTemplate(template: string, values: Record<string, string>): string {
	TEMPLATE_EXPRESSION_REGEX.lastIndex = 0;

	return template.replace(
		TEMPLATE_EXPRESSION_REGEX,
		(_match, operator: string, varList: string) => {
			const varNames = varList
				.split(',')
				.map((v: string) =>
					v
						.replace(VARIABLE_EXPLODE_MODIFIER_REGEX, '')
						.replace(VARIABLE_PREFIX_MODIFIER_REGEX, '')
						.trim()
				);

			const expandedParts = varNames
				.map((name: string) => values[name] ?? '')
				.filter((v: string) => v !== '');

			if (expandedParts.length === 0) return '';

			switch (operator) {
				case URI_TEMPLATE_OPERATORS.RESERVED:
					// Reserved expansion: no encoding
					return expandedParts.join(URI_TEMPLATE_SEPARATORS.COMMA);
				case URI_TEMPLATE_OPERATORS.FRAGMENT:
					// Fragment expansion
					return (
						URI_TEMPLATE_OPERATORS.FRAGMENT + expandedParts.join(URI_TEMPLATE_SEPARATORS.COMMA)
					);
				case URI_TEMPLATE_OPERATORS.PATH_SEGMENT:
					// Path segments
					return URI_TEMPLATE_SEPARATORS.SLASH + expandedParts.join(URI_TEMPLATE_SEPARATORS.SLASH);
				case URI_TEMPLATE_OPERATORS.LABEL:
					// Label expansion
					return (
						URI_TEMPLATE_SEPARATORS.PERIOD + expandedParts.join(URI_TEMPLATE_SEPARATORS.PERIOD)
					);
				case URI_TEMPLATE_OPERATORS.PATH_PARAM:
					// Path-style parameters
					return varNames
						.filter((_: string, i: number) => expandedParts[i])
						.map(
							(name: string, i: number) =>
								`${URI_TEMPLATE_SEPARATORS.SEMICOLON}${name}=${expandedParts[i]}`
						)
						.join('');
				case URI_TEMPLATE_OPERATORS.FORM_QUERY:
					// Form-style query
					return (
						URI_TEMPLATE_SEPARATORS.QUERY_PREFIX +
						varNames
							.filter((_: string, i: number) => expandedParts[i])
							.map(
								(name: string, i: number) =>
									`${encodeURIComponent(name)}=${encodeURIComponent(expandedParts[i])}`
							)
							.join(URI_TEMPLATE_SEPARATORS.QUERY_CONTINUATION)
					);
				case URI_TEMPLATE_OPERATORS.FORM_CONTINUATION:
					// Form-style query continuation
					return (
						URI_TEMPLATE_SEPARATORS.QUERY_CONTINUATION +
						varNames
							.filter((_: string, i: number) => expandedParts[i])
							.map(
								(name: string, i: number) =>
									`${encodeURIComponent(name)}=${encodeURIComponent(expandedParts[i])}`
							)
							.join(URI_TEMPLATE_SEPARATORS.COMMA)
					);
				default:
					// Simple string expansion (default operator)
					return expandedParts
						.map((v: string) => encodeURIComponent(v))
						.join(URI_TEMPLATE_SEPARATORS.COMMA);
			}
		}
	);
}

/**
 * Check whether all required variables in a template have been provided.
 *
 * @param template - URI template string
 * @param values - Map of variable name to value
 * @returns true if all variables have non-empty values
 */
export function isTemplateComplete(template: string, values: Record<string, string>): boolean {
	const variables = extractTemplateVariables(template);

	return variables.every((v) => (values[v.name] ?? '').trim() !== '');
}
