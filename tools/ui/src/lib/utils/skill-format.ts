/**
 * SKILL.md-compatible serialization helpers for skills.
 *
 * Each skill is exported as a single markdown file with YAML frontmatter,
 * matching the Agent Skills specification
 * (https://agentskills.io/specification). Pi's skill loader reads the same
 * layout directly, so an exported llama-ui skill round-trips into Pi's
 * `~/.pi/agent/skills/` directory without translation.
 *
 * File layout (per skill):
 *
 *   ---
 *   name: my-skill
 *   description: One-line summary suitable for system-prompt injection.
 *   ---
 *
 *   <markdown body — the skill's instruction text>
 *
 * Frontmatter is intentionally minimal: top-level scalars only. Anything
 * more exotic (nested maps, lists, multiline strings) is supported by Pi
 * and can be layered on by a future Skills PR without breaking the
 * round-trip.
 *
 * Name validation follows the Agent Skills spec:
 *   - 1–64 characters
 *   - Lowercase letters, numbers, hyphens only
 *   - No leading/trailing hyphens
 *   - No consecutive hyphens
 */

import type { DatabaseSkill } from '$lib/types';
import { parseFrontmatter, serializeFrontmatter } from './frontmatter';

/** YAML-frontmatter keys this codebase emits / parses back (Agent Skills spec). */
export const SKILL_FRONTMATTER_KEYS = [
	'name',
	'description',
	'license',
	'compatibility',
	'metadata',
	'allowed-tools',
	'disable-model-invocation'
] as const;

/** Optional fields supported by the Agent Skills specification. */
export type SkillFrontmatter = {
	name?: string;
	description?: string;
	license?: string;
	compatibility?: string;
	metadata?: string;
	'allowed-tools'?: string;
	'disable-model-invocation'?: boolean;
};

/**
 * Normalize a skill name to conform with the Agent Skills standard.
 *
 * Rules applied:
 *   - Lowercase
 *   - Replace invalid characters with hyphens
 *   - Collapse consecutive hyphens
 *   - Strip leading/trailing hyphens
 */
export function normalizeSkillName(name: string): string {
	return name
		.toLowerCase()
		.replace(/[^a-z0-9-]/g, '-')
		.replace(/-+/g, '-')
		.replace(/^-|-$/g, '');
}

/**
 * Validate a skill name against the Agent Skills standard.
 *
 * @returns null if valid, or an error message string if invalid.
 */
export function validateSkillName(name: string): string | null {
	if (!name || !name.trim()) return 'Name is required';
	if (name.length > 64) return 'Name must be ≤ 64 characters';
	if (!/^[a-z0-9]$/.test(name) && !/^[a-z0-9][a-z0-9-]*[a-z0-9]$/.test(name)) {
		return 'Name must be lowercase, contain only a-z, 0-9, and hyphens, with no leading/trailing hyphens';
	}
	if (/--/.test(name)) return 'Name must not contain consecutive hyphens';
	return null;
}

/**
 * Validate a skill description against the Agent Skills standard.
 *
 * Description is required on `DatabaseSkill`. Empty / whitespace strings
 * still fail this validator — they exist so the in-form error reflects
 * the user's most recent input before the field has any value at all.
 *
 * @returns null if valid, or an error message string if invalid.
 */
export function validateSkillDescription(desc: string): string | null {
	if (!desc || !desc.trim()) return 'Description is required';
	if (desc.length > 1024) return 'Description must be ≤ 1024 characters';
	return null;
}

/**
 * Serialize one skill as markdown with YAML frontmatter.
 *
 * The `name` is normalized to comply with the Agent Skills spec.
 * `id`, `path`, and `origin` are llama-ui internal fields and are not
 * emitted as frontmatter — they round-trip through the archive's
 * directory layout (`<name>/SKILL.md`) and through `SkillProvider.path`,
 * but they are not part of the Agent Skills spec and would pollute
 * the on-disk format for non-llama-ui consumers.
 *
 * `last-modified` IS emitted as a private convention under the
 * `last-modified` key, used by llama-ui to preserve relative ordering
 * across export/import. Pi's loader ignores unknown frontmatter keys.
 */
export function skillToMarkdown(
	skill: Pick<DatabaseSkill, 'name' | 'content' | 'description'> &
		Partial<
			Pick<
				DatabaseSkill,
				'license' | 'compatibility' | 'metadata' | 'allowedTools' | 'disableModelInvocation'
			> & { id?: string; lastModified?: number; path?: string }
		>
): string {
	const fields: Record<string, string | boolean | number | undefined> = {
		name: normalizeSkillName(skill.name),
		description: skill.description.trim()
	};

	// Emit optional Agent Skills fields only when present
	if (skill.license) fields.license = skill.license;
	if (skill.compatibility) fields.compatibility = skill.compatibility;
	if (skill.metadata !== undefined) {
		fields.metadata =
			typeof skill.metadata === 'string' ? skill.metadata : JSON.stringify(skill.metadata);
	}
	if (skill.allowedTools) fields['allowed-tools'] = skill.allowedTools;
	if (skill.disableModelInvocation !== undefined)
		fields['disable-model-invocation'] = !!skill.disableModelInvocation;

	// Private llama-ui marker; Pi's loader treats unknown keys benignly.
	if (skill.lastModified !== undefined) fields['last-modified'] = skill.lastModified;

	return serializeFrontmatter(fields, skill.content ?? '');
}

/**
 * Deserialize a single markdown file back into a skill row.
 *
 * The returned object has only the user-editable fields; the caller is
 * responsible for assigning a fresh `id` and `lastModified` if those are
 * missing (e.g. on first import). `origin` defaults to `'lib'` — the
 * caller may override once a `path` is known so filesystem provenance
 * is preserved.
 *
 * `description` is always returned as a string (possibly empty); the
 * store rejects empty values via `validateSkillDescription`. Callers
 * that want a strict import should also gate on a non-empty description
 * before calling `addSkill`.
 */
export function markdownToSkill(
	raw: string,
	existingId?: string
): Omit<DatabaseSkill, 'id' | 'lastModified' | 'origin'> & {
	id?: string;
	lastModified?: number;
} {
	const { frontmatter, body } = parseFrontmatter(raw);
	const fm = frontmatter as Record<string, unknown>;

	const name = stringifyScalar(fm.name).trim();
	const description = stringifyScalar(fm.description).trim() || '';

	// Extract optional Agent Skills fields
	const license = fm.license ? stringifyScalar(fm.license) || undefined : undefined;
	const compatibility = fm.compatibility
		? stringifyScalar(fm.compatibility) || undefined
		: undefined;
	const metadata = fm.metadata
		? typeof fm.metadata === 'string'
			? fm.metadata
			: JSON.stringify(fm.metadata)
		: undefined;
	const allowedTools = fm['allowed-tools']
		? stringifyScalar(fm['allowed-tools']) || undefined
		: undefined;
	const disableModelInvocation =
		fm['disable-model-invocation'] !== undefined && fm['disable-model-invocation'] !== null
			? !!fm['disable-model-invocation']
			: undefined;

	// Parse lastModified from frontmatter (internal field, not part of spec)
	let lastModified: number | undefined;
	const rawTs = fm['last-modified'];
	if (typeof rawTs === 'number') {
		lastModified = rawTs;
	} else {
		const tsString = stringifyScalar(rawTs).trim();
		if (tsString) {
			const n = Number(tsString);
			if (!Number.isNaN(n)) lastModified = n;
		}
	}

	return {
		id: existingId,
		name,
		description,
		content: body,
		...(license && { license }),
		...(compatibility && { compatibility }),
		...(metadata && { metadata }),
		...(allowedTools && { allowedTools }),
		...(disableModelInvocation !== undefined && { disableModelInvocation }),
		...(lastModified && !Number.isNaN(lastModified) ? { lastModified } : {})
	};
}

function stringifyScalar(value: unknown): string {
	if (typeof value === 'string') return value;
	if (value === undefined || value === null) return '';
	return String(value);
}

/**
 * Compose the system-prompt contribution for a set of always-on skills.
 *
 * Pure helper with no dependencies on the chat store: takes a base
 * prompt and the list of always-on skill rows (already filtered by the
 * caller against `skillPreferencesStore`), returns the composed text
 * plus the same list so the caller can stamp `DatabaseMessageExtraSkill`
 * extras onto the system message for the chat UI.
 *
 * The output is intentionally plain text — just the concatenated
 * skill contents joined by blank lines. The system message body must
 * stay a faithful byte-for-byte representation of the underlying skill
 * content so that the "Sync from library" flow can compare it against
 * the source row, and so that the edit form never shows injected
 * scaffolding. The skill's name and description are *not* included in
 * the body; they live on the `DatabaseMessageExtraSkill` extras that
 * the chat UI uses to render the skill card.
 *
 * Skills with empty `content` are skipped entirely. Skills with
 * `disableModelInvocation: true` are also skipped — that flag is part
 * of the Agent Skills spec and means the user has opted the skill out
 * of agent invocation.
 */
export function composeSystemPromptWithAlwaysOnSkills<
	T extends Pick<DatabaseSkill, 'id' | 'name' | 'content' | 'disableModelInvocation'>
>(baseSystemPrompt: string, alwaysOnSkills: T[]): { text: string; skills: T[] } {
	const included = alwaysOnSkills.filter(
		(s) => !s.disableModelInvocation && s.content.trim().length > 0
	);
	if (included.length === 0) return { text: baseSystemPrompt, skills: [] };

	const joined = included.map((s) => s.content.trim()).join('\n\n');
	const text = baseSystemPrompt ? `${baseSystemPrompt}\n\n${joined}` : joined;
	return { text, skills: included };
}
