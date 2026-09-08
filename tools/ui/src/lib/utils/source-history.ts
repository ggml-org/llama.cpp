/**
 * Source-space undo/redo history for the ChatFormInputRich, whose
 * imperative DOM rebuilds destroy the browser's native undo stack.
 * Entries record the state BEFORE an edit; edits within `groupWindowMs`
 * extend the open group so a typing burst undoes as a unit, while
 * structural edits (paste, mention insert, clear) pass `newGroup`.
 */

export interface SourceHistoryEntry {
	value: string;
	caret: number;
}

export class SourceHistory {
	private lastPush = 0;
	private redoStack: SourceHistoryEntry[] = [];
	private undoStack: SourceHistoryEntry[] = [];

	constructor(
		private limit = 100,
		private groupWindowMs = 800
	) {}

	push(entry: SourceHistoryEntry, now: number, newGroup = false): void {
		if (newGroup || now - this.lastPush >= this.groupWindowMs || this.undoStack.length === 0) {
			this.undoStack.push(entry);

			if (this.undoStack.length > this.limit) this.undoStack.shift();
		}

		this.lastPush = now;
		this.redoStack = [];
	}

	redo(current: SourceHistoryEntry): SourceHistoryEntry | null {
		const entry = this.redoStack.pop();

		if (!entry) return null;

		this.undoStack.push(current);
		this.lastPush = 0;

		return entry;
	}

	undo(current: SourceHistoryEntry): SourceHistoryEntry | null {
		const entry = this.undoStack.pop();

		if (!entry) return null;

		this.redoStack.push(current);
		this.lastPush = 0; // the next edit after an undo starts a new group

		return entry;
	}
}
