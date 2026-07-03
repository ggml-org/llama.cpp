/**
 * Bulk skill archive helpers (zip of `<name>/SKILL.md` files).
 *
 * The archive layout mirrors Pi's on-disk skill layout:
 *
 *   llama_skills_<date>.zip
 *   ├── code-reviewer/SKILL.md
 *   ├── python-tutor/SKILL.md
 *   └── ...
 *
 * Each `SKILL.md` is the same output as `skillToMarkdown` — so the archive
 * can be unzipped straight into `~/.pi/agent/skills/` (or any other
 * directory Pi scans) and Pi will load it natively.
 */

import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';
import { normalizeSkillName, skillToMarkdown, markdownToSkill } from './skill-format';
import type { DatabaseSkill } from '$lib/types';

export type SkillFiles = Record<string, string>;

/**
 * Build the `<name>/SKILL.md` file set for a single skill. The directory
 * name is the skill's name — that's the layout Pi expects. Future
 * Skills PRs (with auxiliary resource files) extend this same helper.
 */
export function skillToFiles(
	skill: Pick<DatabaseSkill, 'name' | 'content' | 'description'> & Partial<DatabaseSkill>
): SkillFiles {
	return { 'SKILL.md': skillToMarkdown(skill) };
}

/**
 * Pack an array of skills into a zip Blob, one folder per skill.
 *
 * Skill names are normalized using Agent Skills spec rules to avoid
 * zip path-traversal issues. If two skills collide on the same
 * normalized name a warning is logged and a numeric suffix is appended.
 */
export async function skillsToArchive(skills: DatabaseSkill[]): Promise<Blob> {
	const seen = new Set<string>();
	const entries: Record<string, Uint8Array> = {};
	for (const skill of skills) {
		const dirName = normalizeSkillName(skill.name);
		const uniqueDirName = uniqueName(dirName, seen);
		if (uniqueDirName !== dirName && !seen.has(dirName)) {
			// Don't double-warn on identical collisions.
			console.warn(
				`[skill-archive] Skill name "${skill.name}" was sanitized to "${uniqueDirName}" for archive path safety`
			);
		}
		entries[`${uniqueDirName}/SKILL.md`] = strToU8(skillToMarkdown(skill));
	}
	const zipped = zipSync(entries, { level: 6 });
	return new Blob([zipped], { type: 'application/zip' });
}

/**
 * Unpack a zip Blob (or raw `Uint8Array`) into individual SKILL.md
 * payloads, dropping anything that does not live in a `…/SKILL.md`
 * directory entry.
 */
export async function archiveToSkills(input: Blob | Uint8Array): Promise<string[]> {
	const bytes = input instanceof Blob ? new Uint8Array(await input.arrayBuffer()) : input;
	const entries = unzipSync(bytes);
	const out: string[] = [];
	for (const [path, data] of Object.entries(entries)) {
		if (!path.endsWith('/SKILL.md')) continue;
		// Skip path-traversal entries like `../foo/SKILL.md`.
		if (path.includes('..')) continue;
		out.push(strFromU8(data));
	}
	return out;
}

/**
 * Parse every recovered SKILL.md into a partial skill row plus its
 * optional `id` (taken from the parent folder name when it looks like
 * a UUID).
 */
export async function archiveToSkillRows(input: Blob | Uint8Array): Promise<
	Array<
		Omit<DatabaseSkill, 'id' | 'lastModified' | 'origin'> & {
			id?: string;
			lastModified?: number;
			origin?: DatabaseSkill['origin'];
		}
	>
> {
	const bytes = input instanceof Blob ? new Uint8Array(await input.arrayBuffer()) : input;
	const entries = unzipSync(bytes);
	const out: Array<
		Omit<DatabaseSkill, 'id' | 'lastModified' | 'origin'> & {
			id?: string;
			lastModified?: number;
			origin?: DatabaseSkill['origin'];
		}
	> = [];

	for (const [path, data] of Object.entries(entries)) {
		if (!path.endsWith('/SKILL.md')) continue;
		if (path.includes('..')) continue;
		const parentName = path.split('/').slice(-2, -1)[0] ?? '';
		const text = strFromU8(data);
		const existingId = isUuid(parentName) ? parentName : undefined;
		out.push(markdownToSkill(text, existingId));
	}
	return out;
}

function uniqueName(base: string, taken: Set<string>): string {
	if (!taken.has(base)) {
		taken.add(base);
		return base;
	}
	let i = 2;
	while (taken.has(`${base}-${i}`)) i++;
	const next = `${base}-${i}`;
	taken.add(next);
	return next;
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
function isUuid(s: string): boolean {
	return UUID_RE.test(s);
}
