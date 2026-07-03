import { browser } from '$app/environment';
import { DatabaseService } from '$lib/services/database.service';
import {
	skillToMarkdown,
	markdownToSkill,
	skillsToArchive,
	archiveToSkillRows,
	validateSkillDescription
} from '$lib/utils';
import { skillPreferencesStore } from './skill-preferences.svelte';
import type { SkillOrigin, Skill } from '$lib/types';

/** Result of a markdown / archive import. */
export type ImportResult = {
	added: number;
	updated: number;
	skipped: number;
};

class SkillsStore {
	/** In-memory cache — source of truth for the UI; persisted to IndexedDB. */
	#items = $state<Skill[]>([]);

	constructor() {
		this.#loadFromDb();
	}

	async #loadFromDb(): Promise<void> {
		if (!browser) return;
		try {
			const rows = await DatabaseService.getAllSkills();
			// Drop any rows that don't satisfy the current schema. Removed rows are
			// not re-persisted; clear browser storage to fully reset (start of
			// day, before any user has shipped data through this feature).
			const valid = rows.filter(isValidSkill);
			const dropped = rows.length - valid.length;
			if (dropped > 0) {
				console.warn(
					`[SkillsStore] Ignored ${dropped} skill row(s) that do not match the current schema; clear browser storage to fully reset.`
				);
			}
			// Coerce missing `origin` to `'lib'` for legacy rows that were
			// authored before the SkillOrigin split. We mutate the snapshot
			// in place so the rest of the store sees a uniform shape.
			//
			// One-shot migration: rows written before the prefs-store split
			// carried `alwaysOn: true` on the row itself. Move those flags
			// into `skillPreferencesStore` so the rest of the app sees a
			// uniform, prefs-driven shape. `alwaysOn` is no longer stored
			// on the skill; legacy rows keep working transparently.
			let migrated = 0;
			const migratedIds: string[] = [];
			const normalized = valid.map((s) => {
				const withoutAlwaysOn = { ...s };
				const legacyAlwaysOn = (withoutAlwaysOn as unknown as { alwaysOn?: boolean }).alwaysOn;
				delete (withoutAlwaysOn as unknown as { alwaysOn?: boolean }).alwaysOn;
				const withOrigin = withoutAlwaysOn.origin
					? withoutAlwaysOn
					: { ...withoutAlwaysOn, origin: 'lib' as SkillOrigin };
				if (legacyAlwaysOn) {
					migratedIds.push(withOrigin.id);
					migrated++;
				}
				return withOrigin;
			});
			for (const id of migratedIds) skillPreferencesStore.setAlwaysOn(id, true);
			if (migrated > 0) {
				console.warn(
					`[SkillsStore] Migrated ${migrated} always-on skill flag(s) from skill rows to skillPreferencesStore`
				);
			}
			this.#items = normalized;
		} catch (error) {
			console.warn('[SkillsStore] Failed to load from IndexedDB:', error);
		}
	}

	getSkills(): Skill[] {
		return [...this.#items].sort((a, b) => b.lastModified - a.lastModified);
	}

	getSkill(id: string): Skill | undefined {
		return this.#items.find((i) => i.id === id);
	}

	async addSkill(
		skill: Omit<Skill, 'id' | 'lastModified' | 'origin'> & {
			id?: string;
			path?: string;
			origin?: SkillOrigin;
		}
	): Promise<Skill> {
		// Description is required (Agent Skills spec + llama-ui always-on gate).
		const description = (skill.description ?? '').trim();
		const descriptionError = validateSkillDescription(description);
		if (descriptionError) throw new Error(descriptionError);

		// Derive a stable row id from `path` when one is provided so
		// filesystem-sourced skills keep their identity across reloads.
		// Otherwise fall back to a UUID for in-app authored rows.
		const id = skill.id?.trim() || skill.path?.trim() || crypto.randomUUID();
		const origin: SkillOrigin = skill.origin ?? (skill.path ? 'user' : 'lib');
		const newSkill: Skill = {
			...skill,
			description,
			id,
			origin,
			lastModified: Date.now()
		};
		this.#items = [newSkill, ...this.#items];
		await DatabaseService.addSkill(newSkill);
		return newSkill;
	}

	async updateSkill(id: string, updates: Partial<Skill>): Promise<Skill | undefined> {
		const idx = this.#items.findIndex((i) => i.id === id);
		if (idx === -1) return undefined;

		const updated: Skill = {
			...this.#items[idx],
			...updates,
			lastModified: Date.now()
		};
		const all = [...this.#items];
		all[idx] = updated;
		this.#items = all;
		await DatabaseService.updateSkill(id, {
			...updates,
			lastModified: updated.lastModified
		});
		return updated;
	}

	async deleteSkill(id: string): Promise<void> {
		this.#items = this.#items.filter((i) => i.id !== id);
		await DatabaseService.deleteSkill(id);
	}

	searchSkills(query: string): Skill[] {
		if (!query.trim()) return this.getSkills();
		const lowerQuery = query.toLowerCase();
		return this.getSkills().filter(
			(s) =>
				s.name.toLowerCase().includes(lowerQuery) ||
				s.content.toLowerCase().includes(lowerQuery) ||
				s.description.toLowerCase().includes(lowerQuery)
		);
	}

	/**
	 * Bulk export — zip of `<name>/SKILL.md` files. The archive is the
	 * same layout Pi loads natively, so it can be unzipped straight into
	 * `~/.pi/agent/skills/` and picked up there.
	 */
	async exportSkillsArchive(): Promise<Blob> {
		const items = $state.snapshot(this.#items) as Skill[];
		return skillsToArchive(items);
	}

	/**
	 * Single-skill export — produces a single `SKILL.md`-shaped markdown
	 * document so a skill can be shared on its own.
	 */
	exportSkillMarkdown(id: string): string | null {
		const skill = this.getSkill(id);
		if (!skill) return null;
		return skillToMarkdown($state.snapshot(skill));
	}

	/**
	 * Bulk import — accepts:
	 *   - a zip archive (one `<name>/SKILL.md` per skill), or
	 *   - a single SKILL.md-shaped markdown blob.
	 *
	 * Matching is by `id` first (when present), then by `name`, so edits
	 * made on another machine still update the same row.
	 */
	async importSkills(file: File | Blob): Promise<ImportResult> {
		const filename = (file as File).name ?? '';
		const rows = await this.#parseImportFile(file, filename);
		if (rows.length === 0) {
			throw new Error('Invalid skills file: no SKILL.md entries found');
		}

		const existingById = new Map(this.#items.map((s) => [s.id, s]));
		const existingByName = new Map(
			this.#items.filter((s) => s.name?.trim()).map((s) => [s.name.trim(), s])
		);

		let added = 0;
		let updated = 0;
		let skipped = 0;

		for (const parsed of rows) {
			if (!parsed.name?.trim() || !parsed.description?.trim() || !parsed.content?.trim()) {
				skipped++;
				continue;
			}

			const existing =
				(parsed.id && existingById.get(parsed.id)) || existingByName.get(parsed.name.trim());

			if (existing) {
				await this.updateSkill(existing.id, {
					name: parsed.name,
					description: parsed.description,
					content: parsed.content,
					license: parsed.license,
					compatibility: parsed.compatibility,
					metadata: parsed.metadata,
					allowedTools: parsed.allowedTools,
					disableModelInvocation: parsed.disableModelInvocation
				});
				updated++;
			} else {
				await this.addSkill({
					name: parsed.name,
					description: parsed.description,
					content: parsed.content,
					path: parsed.path,
					license: parsed.license,
					compatibility: parsed.compatibility,
					metadata: parsed.metadata,
					allowedTools: parsed.allowedTools,
					disableModelInvocation: parsed.disableModelInvocation
				});
				added++;
			}
		}

		return { added, updated, skipped };
	}

	async #parseImportFile(
		file: File | Blob,
		filename: string
	): Promise<
		Array<Omit<DatabaseSkill, 'id' | 'lastModified'> & { id?: string; lastModified?: number }>
	> {
		const lower = filename.toLowerCase();
		if (lower.endsWith('.zip') || file.type === 'application/zip') {
			return archiveToSkillRows(file);
		}
		// Markdown: a single SKILL.md file or our legacy fenced multi-block form.
		const text = await file.text();
		const fencedBlocks = splitLegacyFencedSkillFile(text);
		if (fencedBlocks.length > 0) return fencedBlocks;
		const parsed = markdownToSkill(text);
		return [parsed];
	}
}

interface SkillMarkdownBlock {
	id?: string;
	text: string;
}

const LEGACY_BLOCK_OPEN = /<!-- llama-ui:prompt:(\d+):([0-9a-fA-F-]+) -->/;
const LEGACY_BLOCK_CLOSE_PREFIX = '<!-- /llama-ui:prompt:';

/**
 * Backwards-compatible splitter for the previous prompts-only export
 * format (HTML-comment fences around every record). New exports are
 * zip archives and don't go through this path; we keep the splitter so
 * users with an old single-file backup can still import it.
 */
function splitLegacyFencedSkillFile(raw: string): SkillMarkdownBlock[] {
	const trimmed = raw.trim();
	if (!trimmed || !LEGACY_BLOCK_OPEN.test(trimmed)) return [];

	LEGACY_BLOCK_OPEN.lastIndex = 0;
	const segments: { start: number; end: number; id?: string }[] = [];
	let match: RegExpExecArray | null;
	while ((match = LEGACY_BLOCK_OPEN.exec(trimmed)) !== null) {
		segments.push({ start: match.index + match[0].length, end: -1, id: match[2] });
	}

	const blocks: SkillMarkdownBlock[] = [];
	for (let i = 0; i < segments.length; i++) {
		const seg = segments[i];
		const closeStart = trimmed.indexOf(`${LEGACY_BLOCK_CLOSE_PREFIX}${i} -->`, seg.start);
		seg.end = closeStart === -1 ? trimmed.length : closeStart;
		const body = trimmed.slice(seg.start, seg.end).trim();
		if (body) blocks.push({ id: seg.id, text: body });
	}
	return blocks;
}

export const skillsStore = new SkillsStore();

const VALID_SKILL_ORIGINS: ReadonlySet<SkillOrigin> = new Set<SkillOrigin>([
	'lib',
	'user',
	'project'
]);

/**
 * Lightweight runtime guard for rows coming back from IndexedDB.
 *
 * After the Skills Origin split, indexed rows are expected to carry an
 * `origin` tag (`lib` for in-app, `user` / `project` for filesystem-).
 * Rows that pre-date this field are coerced to `'lib'` instead of being
 * dropped — the user can still see and edit them, they just don't get
 * the new filesystem semantics.
 *
 * Description is required (Agent Skills spec, llama-ui always-on gate).
 * Rows without a description are rejected to prevent silent UX breakage.
 *
 * Legacy rows written before the prefs-store split may still carry an
 * `alwaysOn: true` flag; we accept them here so the migration in
 * `#loadFromDb` can move the flag into `skillPreferencesStore` once
 * at load time, then strip it from the in-memory shape.
 */
function isValidSkill(row: unknown): row is Skill & { alwaysOn?: boolean } {
	if (!row || typeof row !== 'object') return false;
	const s = row as Partial<Skill>;
	if (
		typeof s.id !== 'string' ||
		typeof s.name !== 'string' ||
		s.name.trim().length === 0 ||
		typeof s.description !== 'string' ||
		s.description.trim().length === 0 ||
		typeof s.content !== 'string' ||
		typeof s.lastModified !== 'number'
	) {
		return false;
	}
	if (s.path !== undefined && typeof s.path !== 'string') return false;
	if (s.origin !== undefined && !VALID_SKILL_ORIGINS.has(s.origin as SkillOrigin)) return false;
	return true;
}
