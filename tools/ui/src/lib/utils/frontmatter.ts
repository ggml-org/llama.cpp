/**
 * Lightweight YAML frontmatter parser for SKILL.md-style files.
 *
 * Used to round-trip Prompt / Skill files with the format documented at
 * https://agentskills.io/specification. We intentionally do not depend on
 * a full YAML library — the frontmatter we emit and parse is a tiny subset:
 * top-level strings + booleans + numeric-looking numbers (so timestamps
 * round-trip as integers, not strings).
 *
 * Format example (the parser only inspects the part between the leading
 * `---` and the next `\n---` or end-of-string):
 *
 *   ---
 *   name: My Prompt
 *   description: A short summary
 *   last-modified: 1700000000000
 *   ---
 *
 *   Markdown body here.
 */

export interface FrontmatterParseResult<T = Record<string, unknown>> {
	/** Parsed top-level frontmatter keys. */
	frontmatter: Partial<T>;
	/** Body content with the frontmatter block removed. */
	body: string;
}

const FRONTMATTER_OPEN = /^---\r?\n/;
// Matches `\n---` (closing fence) without anchoring to end-of-string so we
// detect the close even when a body follows.
const FRONTMATTER_CLOSE = /\r?\n---(?:\r?\n|$)/;

/**
 * Parse a markdown string with optional YAML frontmatter.
 * Returns the body and the parsed top-level keys.
 */
export function parseFrontmatter<T = Record<string, unknown>>(
	content: string
): FrontmatterParseResult<T> {
	const normalized = content.replace(/\r\n/g, '\n');

	if (!normalized.startsWith('---')) {
		return { frontmatter: {} as Partial<T>, body: normalized };
	}

	const afterOpen = normalized.replace(FRONTMATTER_OPEN, '');
	const closeMatch = afterOpen.match(FRONTMATTER_CLOSE);
	if (!closeMatch) {
		// Unterminated frontmatter — treat whole document as body
		return { frontmatter: {} as Partial<T>, body: normalized };
	}

	const endOfYaml = closeMatch.index ?? 0;
	const endOfFence = endOfYaml + closeMatch[0].length;
	const yamlString = afterOpen.slice(0, endOfYaml);
	const body = afterOpen.slice(endOfFence).replace(/^\n+/, '');

	return { frontmatter: parseSimpleYaml(yamlString) as Partial<T>, body };
}

/**
 * Serialize a flat key/value record as YAML frontmatter + body.
 *
 * Scalar values are emitted unquoted when safe; values containing
 * YAML-special characters (`:`, `#`, leading `-`, `{`, `}`, `[`, `]`,
 * leading/trailing whitespace, or strings that look like `true` / `false` /
 * `null` / numbers) are wrapped in double quotes and escaped.
 */
export function serializeFrontmatter(
	fields: Record<string, string | boolean | number | undefined>,
	body: string
): string {
	const lines: string[] = ['---'];
	for (const [key, value] of Object.entries(fields)) {
		if (value === undefined) continue;
		if (typeof value === 'boolean') {
			lines.push(`${key}: ${value ? 'true' : 'false'}`);
			continue;
		}
		if (typeof value === 'number') {
			lines.push(`${key}: ${value}`);
			continue;
		}
		lines.push(`${key}: ${formatYamlScalar(value)}`);
	}
	lines.push('---', '');
	return lines.join('\n') + body;
}

/**
 * Minimal YAML reader: top-level string / boolean / number scalars only.
 * Anything more exotic (lists, comments, multiline strings, nested maps)
 * is intentionally not supported to keep the parser tiny — we never emit
 * such content ourselves.
 */
function parseSimpleYaml(yaml: string): Record<string, unknown> {
	const result: Record<string, unknown> = {};
	const lines = yaml.split('\n');

	for (const line of lines) {
		if (!line.trim() || line.trim().startsWith('#')) continue;

		const match = line.match(/^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)$/);
		if (!match) continue;

		result[match[1]] = coerceScalar(match[2]);
	}

	return result;
}

function coerceScalar(raw: string): string | boolean | number {
	const trimmed = raw.trim().replace(/^["']|["']$/g, '');
	if (trimmed === 'true') return true;
	if (trimmed === 'false') return false;
	if (trimmed === 'null' || trimmed === '') return '';
	if (/^-?\d+(\.\d+)?$/.test(trimmed)) {
		const asNumber = Number(trimmed);
		if (!Number.isNaN(asNumber)) return asNumber;
	}
	return unquoteYamlScalar(raw);
}

function unquoteYamlScalar(raw: string): string {
	const trimmed = raw.trim();
	if (
		(trimmed.startsWith('"') && trimmed.endsWith('"')) ||
		(trimmed.startsWith("'") && trimmed.endsWith("'"))
	) {
		return trimmed.slice(1, -1);
	}
	return trimmed;
}

function formatYamlScalar(value: string): string {
	// We emit simple `key: value` flow scalars. Quoting is required when
	// the value would otherwise be parsed as something else (number,
	// bool, struct indicator, block scalar) or when it would split on
	// whitespace. We deliberately leave most punctuation alone — YAML
	// flow scalars can carry `>`, `<`, `+`, `-`, `/`, `.`, `,` etc.
	const needsQuoting =
		value === '' ||
		/^(true|false|null|yes|no)$/i.test(value) ||
		/^-?\d+(\.\d+)?$/.test(value) ||
		/^[:#&*?|>%@`]/.test(value) ||
		/[:#]\s/.test(value) ||
		/^\s|\s$/.test(value) ||
		/^-/.test(value);
	if (!needsQuoting) return value;

	const escaped = value.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
	return `"${escaped}"`;
}
