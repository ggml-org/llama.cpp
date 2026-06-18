import { browser } from '$app/environment';
import { INSTRUCTIONS_LOCALSTORAGE_KEY } from '$lib/constants/storage';

export interface Instruction {
	id: string;
	title: string;
	content: string;
	lastModified: number;
}

class InstructionsStore {
	/** Instructions list — persists to localStorage */
	#items = $state<Instruction[]>([]);

	/** Load instructions from localStorage */
	#loadFromStorage(): void {
		if (!browser) return;
		try {
			const raw = localStorage.getItem(INSTRUCTIONS_LOCALSTORAGE_KEY);
			if (raw) {
				const parsed = JSON.parse(raw);
				if (Array.isArray(parsed)) {
					this.#items = parsed;
				}
			}
		} catch (error) {
			console.warn('[InstructionsStore] Failed to load from localStorage:', error);
		}
	}

	/** Persist instructions to localStorage */
	#saveToStorage(): void {
		if (!browser) return;
		try {
			localStorage.setItem(INSTRUCTIONS_LOCALSTORAGE_KEY, JSON.stringify(this.#items));
		} catch (error) {
			console.warn('[InstructionsStore] Failed to save to localStorage:', error);
		}
	}

	// Expose private state for initialization
	get items(): Instruction[] {
		return this.#items;
	}
	set items(value: Instruction[]) {
		this.#items = value;
		this.#saveToStorage();
	}

	constructor() {
		this.#loadFromStorage();
	}

	getInstructions(): Instruction[] {
		return [...this.#items].sort((a, b) => b.lastModified - a.lastModified);
	}

	getInstruction(id: string): Instruction | undefined {
		return this.#items.find((i) => i.id === id);
	}

	addInstruction(instruction: Omit<Instruction, 'id' | 'lastModified'>): Instruction {
		const newInstruction: Instruction = {
			...instruction,
			id: crypto.randomUUID(),
			lastModified: Date.now()
		};
		this.items = [newInstruction, ...this.#items];
		return newInstruction;
	}

	updateInstruction(id: string, updates: Partial<Instruction>): Instruction | undefined {
		const idx = this.#items.findIndex((i) => i.id === id);
		if (idx === -1) return undefined;

		const updated: Instruction = {
			...this.#items[idx],
			...updates,
			lastModified: Date.now()
		};
		const all = [...this.#items];
		all[idx] = updated;
		this.items = all;
		return updated;
	}

	deleteInstruction(id: string): void {
		this.items = this.#items.filter((i) => i.id !== id);
	}

	searchInstructions(query: string): Instruction[] {
		if (!query.trim()) return this.getInstructions();

		const lowerQuery = query.toLowerCase();
		return this.#items.filter(
			(i) =>
				i.title.toLowerCase().includes(lowerQuery) ||
				i.content.toLowerCase().includes(lowerQuery)
		);
	}
}

export const instructionsStore = new InstructionsStore();
