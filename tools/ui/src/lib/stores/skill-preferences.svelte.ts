/**
 * Skill preferences store — localStorage-backed flags that control how
 * llama-ui uses a skill (e.g. "include in every new conversation").
 *
 * Why this is separate from the skills library:
 *
 *   Skills library (IndexedDB, `skillsStore`)  →  canonical Agent Skills
 *     payload. Same bytes you'd find in `~/.pi/agent/skills/<name>/SKILL.md`
 *     or the Agent Skills spec example. Stored verbatim so an export is a
 *     faithful, interoperable snapshot.
 *
 *   Preferences (localStorage, this store)    →  llama-ui UI configuration.
 *     "Always include this skill in the system prompt", future toggles
 *     like "pin in quick-insert", etc. None of these flags leak into
 *     `skill.content`; the chat composer reads them at injection time and
 *     composes the system prompt accordingly.
 *
 * This split also keeps the system-message body clean: no `<skill>`
 * scaffolding, no `<!-- llama-ui:... -->` region markers. The model sees
 * pure skill content. The chat UI renders skill cards because the
 * message's `extra` field carries `DatabaseMessageExtraSkill` records
 * stamped at conversation-create time.
 */

const STORAGE_KEY = 'llama-ui:skill-preferences:v1';

export interface SkillPreferences {
	/** Skill ids marked `alwaysOn` — auto-inject into new conversations. */
	alwaysOn: string[];
}

class SkillPreferencesStore {
	#alwaysOn = $state<Set<string>>(new Set());

	constructor() {
		this.#hydrate();
	}

	#hydrate(): void {
		if (typeof localStorage === 'undefined') return;
		try {
			const raw = localStorage.getItem(STORAGE_KEY);
			if (!raw) return;
			const parsed = JSON.parse(raw) as Partial<SkillPreferences>;
			if (Array.isArray(parsed.alwaysOn)) {
				this.#alwaysOn = new Set(parsed.alwaysOn.filter((s): s is string => typeof s === 'string'));
			}
		} catch (error) {
			console.warn('[SkillPreferencesStore] Failed to parse stored preferences:', error);
		}
	}

	#persist(): void {
		if (typeof localStorage === 'undefined') return;
		const payload: SkillPreferences = { alwaysOn: [...this.#alwaysOn] };
		try {
			localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
		} catch (error) {
			console.warn('[SkillPreferencesStore] Failed to persist preferences:', error);
		}
	}

	/** Whether `id` is marked always-on. */
	isAlwaysOn(id: string): boolean {
		return this.#alwaysOn.has(id);
	}

	/** Snapshot of always-on skill ids. */
	getAlwaysOnIds(): string[] {
		return [...this.#alwaysOn];
	}

	/** Set the always-on flag for a skill. */
	setAlwaysOn(id: string, value: boolean): void {
		const had = this.#alwaysOn.has(id);
		if (value && !had) {
			this.#alwaysOn.add(id);
			this.#persist();
		} else if (!value && had) {
			this.#alwaysOn.delete(id);
			this.#persist();
		}
	}

	/** Toggle the always-on flag. Returns the new value. */
	toggleAlwaysOn(id: string): boolean {
		const next = !this.#alwaysOn.has(id);
		this.setAlwaysOn(id, next);
		return next;
	}
}

export const skillPreferencesStore = new SkillPreferencesStore();
