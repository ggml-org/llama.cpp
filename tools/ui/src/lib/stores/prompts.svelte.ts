import { browser } from '$app/environment';
import { DatabaseService } from '$lib/services/database.service';

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
			this.#items = await DatabaseService.getAllPrompts();
		} catch (error) {
			console.warn('[PromptsStore] Failed to load from IndexedDB:', error);
		}
	}

	getPrompts(category?: string): Prompt[] {
		const items = [...this.#items].sort((a, b) => b.lastModified - a.lastModified);
		if (!category) return items;
		return items.filter((i) => (i.category ?? '') === category);
	}

	getUncategorizedPrompts(): Prompt[] {
		return [...this.#items]
			.sort((a, b) => b.lastModified - a.lastModified)
			.filter((i) => !i.category || !i.category.trim());
	}

	hasUncategorized(): boolean {
		return this.#items.some((i) => !i.category || !i.category.trim());
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

	getCategories(): string[] {
		const categories = new Set(
			this.#items.map((i) => i.category?.trim()).filter((c): c is string => !!c)
		);
		return [...categories].sort((a, b) => a.localeCompare(b));
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
