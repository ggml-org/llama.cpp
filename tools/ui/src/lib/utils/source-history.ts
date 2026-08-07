/**
 * Source-space undo/redo history for the chat-form contenteditable. The
 * component rebuilds its DOM imperatively (replaceChildren), which
 * destroys the browser's native undo stack, so history is kept as
 * (value, caret) snapshots instead.
 *
 * Entries record the state BEFORE an edit. Edits within `groupWindowMs`
 * extend the open group instead of starting a new one, so a typing burst
 * undoes as a unit; structural edits (paste, mention insert, clear) pass
 * `newGroup` to always start one.
 */

export interface SourceHistoryEntry {
	value: string;
	caret: number;
}

export class SourceHistory {
	private undoStack: SourceHistoryEntry[] = [];
	private redoStack: SourceHistoryEntry[] = [];
	private lastPush = 0;

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

	/** Move from `current` to the previous state; null when there is nothing to undo. */
	undo(current: SourceHistoryEntry): SourceHistoryEntry | null {
		const entry = this.undoStack.pop();
		if (!entry) return null;
		this.redoStack.push(current);
		this.lastPush = 0; // the next edit after an undo starts a new group
		return entry;
	}

	/** Move from `current` to the next state; null when there is nothing to redo. */
	redo(current: SourceHistoryEntry): SourceHistoryEntry | null {
		const entry = this.redoStack.pop();
		if (!entry) return null;
		this.undoStack.push(current);
		this.lastPush = 0;
		return entry;
	}
}
