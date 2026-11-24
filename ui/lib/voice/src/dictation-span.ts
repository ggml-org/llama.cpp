import {
	VOICE_CLOSING_PUNCTUATION_PATTERN,
	VOICE_LEADING_SPACE_PATTERN,
	VOICE_TRAILING_SPACE_PATTERN
} from './constants.js';

/**
 * The stretch of the message that dictation writes: it starts at the caret, and each
 * new version of the dictated text replaces only that stretch, so what the user typed
 * before and after it stays. An edit before the stretch moves it; an edit inside it
 * hands that text over to the user, and dictation goes on after it.
 */
export class DictationSpan {
	/**
	 * Edits inside the stretch stay in it, as when its text is a voice message's transcript:
	 * what it held is the user's from then on, and new text goes after it.
	 */
	keepsEdits = false;
	/** Whether the user edited inside the stretch, see keepsEdits. */
	userEdited = false;
	// characters of the dictated text already handed over to the user
	private handedOver = 0;
	// characters at the start of the stretch that the user edited, kept as they are
	private kept = 0;
	private length = 0;
	// characters of the dictated text the stretch shows
	private shown = 0;
	private start: number;

	/** Where the dictated text ends, e.g. to put the caret after it. */
	get end(): number {
		return this.start + this.length;
	}

	/** Whether the stretch shows any text. */
	get isEmpty(): boolean {
		return this.length === 0;
	}

	/**
	 * Writes the dictated text so far into `value` in place of the stretch, with a space
	 * on either side where it would otherwise run into the neighbouring text.
	 */
	apply(value: string, text: string): string {
		const body = text.slice(this.handedOver).trim();
		const before = value.slice(0, this.start + this.kept);
		const after = value.slice(this.start + this.length);
		// no space before text that goes on with punctuation, e.g. ", and then"
		const lead =
			body &&
			before &&
			!VOICE_TRAILING_SPACE_PATTERN.test(before) &&
			!VOICE_CLOSING_PUNCTUATION_PATTERN.test(body)
				? ' '
				: '';
		const trail = body && after && !VOICE_LEADING_SPACE_PATTERN.test(after) ? ' ' : '';
		const inserted = lead + body + trail;

		this.length = this.kept + inserted.length;
		this.shown = text.length;

		return before + inserted + after;
	}

	constructor(value: string, caret: number) {
		this.start = Math.max(0, Math.min(caret, value.length));
	}

	/** Follows an edit the user made, `previous` to `next`. */
	edited(previous: string, next: string): void {
		const { from, insert, to } = changeBetween(previous, next);
		const end = this.end;
		const delta = insert.length - (to - from);

		if (to <= this.start) {
			this.start += delta;
		} else if (from >= end) {
			return;
		} else if (this.keepsEdits && from >= this.start && to <= end) {
			this.length += delta;
			this.kept = this.length;
			this.handedOver = this.shown;
			this.userEdited = true;
		} else {
			// typed into the dictated text: it is theirs now, what comes next goes after it
			this.handedOver = this.shown;
			this.start = to <= end ? end + delta : from + insert.length;
			this.length = 0;
		}
	}

	/** Takes the stretch out of `value`, as a cancelled dictation leaves the message. */
	remove(value: string): string {
		const next = value.slice(0, this.start) + value.slice(this.start + this.length);

		this.length = 0;

		return next;
	}

	/** The text the stretch holds in `value`, with the user's edits inside it. */
	text(value: string): string {
		return value.slice(this.start, this.start + this.length);
	}
}

/** The single change that turns `current` into `next`: their differing middle. */
function changeBetween(current: string, next: string) {
	let from = 0;

	while (from < current.length && from < next.length && current[from] === next[from]) from++;

	let currentTo = current.length;
	let nextTo = next.length;

	while (currentTo > from && nextTo > from && current[currentTo - 1] === next[nextTo - 1]) {
		currentTo--;
		nextTo--;
	}

	return { from, insert: next.slice(from, nextTo), to: currentTo };
}
