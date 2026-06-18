import { browser } from '$app/environment';
import { DatabaseService } from '$lib/services/database.service';
import { INSTRUCTIONS_LOCALSTORAGE_KEY, PROMPTS_LOCALSTORAGE_KEY } from '$lib/constants/storage';

export type Prompt = DatabasePrompt;

class PromptsStore {
	/** In-memory cache - source of truth for the UI; persisted to IndexedDB */
	#items = $state<Prompt[]>([]);

	constructor() {
		this.#loadFromDb();
	}

	async #loadFromDb(): Promise<void> {
		if (!browser) return;
		try {
			const items = await DatabaseService.getAllPrompts();
			this.#items = await this.#migrateFromLocalStorage(items);
		} catch (error) {
			console.warn('[PromptsStore] Failed to load from IndexedDB:', error);
		}
	}

	/** One-time migration from localStorage to IndexedDB (preserves IDs) */
	async #migrateFromLocalStorage(existing: Prompt[]): Promise<Prompt[]> {
		if (!browser) return existing;
		try {
			const raw =
				localStorage.getItem(PROMPTS_LOCALSTORAGE_KEY) ??
				localStorage.getItem(INSTRUCTIONS_LOCALSTORAGE_KEY);
			if (!raw) return existing;

			const parsed = JSON.parse(raw);
			if (!Array.isArray(parsed)) return existing;

			const existingIds = new Set(existing.map((i) => i.id));
			const toMigrate = parsed.filter(
				(i): i is Prompt =>
					i &&
					typeof i.id === 'string' &&
					typeof i.title === 'string' &&
					typeof i.content === 'string' &&
					typeof i.lastModified === 'number' &&
					!existingIds.has(i.id)
			);

			for (const prompt of toMigrate) {
				await DatabaseService.addPrompt(prompt);
			}

			localStorage.removeItem(PROMPTS_LOCALSTORAGE_KEY);
			localStorage.removeItem(INSTRUCTIONS_LOCALSTORAGE_KEY);
			return [...existing, ...toMigrate];
		} catch {
			return existing;
		}
	}

	getPrompts(): Prompt[] {
		return [...this.#items].sort((a, b) => b.lastModified - a.lastModified);
	}

	getPrompt(id: string): Prompt | undefined {
		return this.#items.find((i) => i.id === id);
	}

	async addPrompt(prompt: Omit<Prompt, 'id' | 'lastModified'>): Promise<Prompt> {
		const newPrompt: Prompt = {
			...prompt,
			id: crypto.randomUUID(),
			lastModified: Date.now()
		};
		this.#items = [newPrompt, ...this.#items];
		await DatabaseService.addPrompt(newPrompt);
		return newPrompt;
	}

	async updatePrompt(id: string, updates: Partial<Prompt>): Promise<Prompt | undefined> {
		const idx = this.#items.findIndex((i) => i.id === id);
		if (idx === -1) return undefined;

		const updated: Prompt = {
			...this.#items[idx],
			...updates,
			lastModified: Date.now()
		};
		const all = [...this.#items];
		all[idx] = updated;
		this.#items = all;
		await DatabaseService.updatePrompt(id, {
			...updates,
			lastModified: updated.lastModified
		});
		return updated;
	}

	async deletePrompt(id: string): Promise<void> {
		this.#items = this.#items.filter((i) => i.id !== id);
		await DatabaseService.deletePrompt(id);
	}

	searchPrompts(query: string): Prompt[] {
		if (!query.trim()) return this.getPrompts();

		const lowerQuery = query.toLowerCase();
		return this.#items.filter(
			(i) =>
				i.title.toLowerCase().includes(lowerQuery) || i.content.toLowerCase().includes(lowerQuery)
		);
	}
}

export const promptsStore = new PromptsStore();
