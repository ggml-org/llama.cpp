/**
 * JSON Schema normalization utilities.
 *
 * Used to normalize the `inputSchema` field exposed by MCP servers and other
 * tool sources into a JSON Schema that includes `type` on every property
 * where it can be inferred from the default value. Some upstream servers omit
 * `type` and rely on defaults; downstream consumers (OpenAI-style chat
 * completions, llama-server) perform more reliably when every property carries
 * an explicit type.
 */

function inferTypeFromDefault(value: unknown): string | undefined {
	if (typeof value === 'string') return 'string';
	if (typeof value === 'boolean') return 'boolean';
	if (typeof value === 'number') return Number.isInteger(value) ? 'integer' : 'number';
	if (Array.isArray(value)) return 'array';
	if (value !== null && typeof value === 'object') return 'object';
	return undefined;
}

/**
 * Recursively normalize a JSON Schema object:
 * - infers `type` from `default` for properties / items that omit it
 * - descends into nested `properties` and `items`
 *
 * Returns a new object — does not mutate the input.
 */
export function normalizeJsonSchema(schema: Record<string, unknown>): Record<string, unknown> {
	if (!schema || typeof schema !== 'object') return schema;

	const normalized: Record<string, unknown> = { ...schema };

	if (normalized.properties && typeof normalized.properties === 'object') {
		const props = normalized.properties as Record<string, Record<string, unknown>>;
		const normalizedProps: Record<string, Record<string, unknown>> = {};
		for (const [key, prop] of Object.entries(props)) {
			if (!prop || typeof prop !== 'object') {
				normalizedProps[key] = prop;
				continue;
			}

			const normalizedProp: Record<string, unknown> = { ...prop };

			if (!normalizedProp.type && normalizedProp.default !== undefined) {
				const inferred = inferTypeFromDefault(normalizedProp.default);
				if (inferred) normalizedProp.type = inferred;
			}

			if (normalizedProp.properties) {
				Object.assign(
					normalizedProp,
					normalizeJsonSchema(normalizedProp as Record<string, unknown>)
				);
			}

			if (normalizedProp.items && typeof normalizedProp.items === 'object') {
				normalizedProp.items = normalizeJsonSchema(normalizedProp.items as Record<string, unknown>);
			}

			normalizedProps[key] = normalizedProp;
		}
		normalized.properties = normalizedProps;
	}

	return normalized;
}
