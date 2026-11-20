import { DictationSpan } from './dictation-span.js';

/** The text field dictation writes into. */
export interface DictationTarget {
	getValue: () => string;
	/** Writes the value as dictation changes it; not a user edit. */
	setValue: (value: string) => void;
	getCaretOffset: () => number;
	/** Puts the caret after the dictated text once it is final. */
	setCaretOffset: (offset: number) => void;
}

export interface DictationOptions extends DictationTarget {
	/** Called when what the getters say changes, for a caller that keeps them reactive */
	onChange?: () => void;
	/** The pass over the whole recording failed; the live text stays */
	onError?: (error: unknown) => void;
}

/** What a finished recording hands dictation: see VoiceSessionResult. */
export interface DictationResult {
	/** Text that streamed while recording, so the field shows it already; null when none did */
	liveText: string | null;
	/** The text of the whole recording, null when it is not transcribed */
	transcript: Promise<string> | null;
}

export interface BeginDictationOptions {
	/** Where the text goes, the caret by default */
	at?: number;
	/** Edits inside the dictated text stay in it, see DictationSpan.keepsEdits */
	keepsEdits?: boolean;
}

export type Dictation = ReturnType<typeof createDictation>;

/**
 * Dictation into a text field, without the recording: the text streams in at the caret
 * (write()), the user can type around it and into it as it goes (edited()), and when the
 * recording stops the whole of it may be transcribed once more and replace the live text
 * (finish()), after which the caret goes after it. Plain TypeScript, so any field can take
 * it: start a voice session with `onText: dictation.write` and hand its result to finish().
 */
export function createDictation(options: DictationOptions) {
	let span: DictationSpan | null = null;
	// the value as dictation last left it, to tell the user's edits from its own
	let last = '';
	let refining: Promise<void> | null = null;
	// bumped when a dictation starts or is cancelled, so text still on its way is dropped
	let generation = 0;

	const changed = () => options.onChange?.();

	function write(text: string) {
		if (!span) return;

		last = span.apply(options.getValue(), text);
		options.setValue(last);
	}

	return {
		/** Whether dictation is writing into the field. */
		get active() {
			return span !== null;
		},

		/** Starts writing at the caret, or `at`, replacing any dictation under way. */
		begin(begin: BeginDictationOptions = {}) {
			generation++;
			last = options.getValue();
			span = new DictationSpan(last, begin.at ?? options.getCaretOffset());
			span.keepsEdits = begin.keepsEdits ?? false;
			changed();
		},

		/** Drops the dictated text, as a cancelled recording leaves the field. */
		cancel() {
			generation++;

			if (!span) return;

			if (!span.isEmpty) {
				last = span.remove(options.getValue());
				options.setValue(last);
			}

			span = null;
			changed();
		},

		/** Follows an edit the user typed while dictation writes into the field. */
		edited() {
			const next = options.getValue();

			span?.edited(last, next);
			last = next;
		},

		/** Where the dictated text ends, e.g. to put the caret after it; -1 when none is active. */
		get end() {
			return span?.end ?? -1;
		},

		/**
		 * Ends the dictation with the recording's result: the whole recording's text replaces
		 * the live text (when there is any to replace it with), then the caret goes after it.
		 * Resolves once the text is final.
		 */
		async finish({ liveText, transcript }: DictationResult): Promise<void> {
			const run = generation;
			const dictated = span;

			if (!dictated) return;

			if (transcript) {
				const refined = (async () => {
					try {
						const text = await transcript;

						if (run !== generation) return;

						if (text.trim() || !liveText) write(text);
					} catch (error) {
						if (run === generation) options.onError?.(error);
					}
				})();

				if (liveText) {
					refining = refined;
					changed();
				}

				await refined;

				if (refining === refined) {
					refining = null;
					changed();
				}
			}

			if (run !== generation) return;

			span = null;
			changed();

			if (!dictated.isEmpty) options.setCaretOffset(dictated.end);
		},

		/** The live text is still being replaced by a pass over the whole recording. */
		get refining() {
			return refining !== null;
		},

		/** Resolves once the text in the field is final, at once when nothing is refining it. */
		settled(): Promise<void> {
			return refining ?? Promise.resolve();
		},

		/**
		 * Takes the dictated text back out of the field and ends the dictation, as a voice
		 * message's live text goes with its clip. Returns the text, and whether the user
		 * edited it; null when there was none.
		 */
		take(): { text: string; userEdited: boolean } | null {
			const taken = span;

			generation++;
			span = null;

			if (!taken) return null;

			changed();

			if (taken.isEmpty) return null;

			const previous = options.getValue();
			const text = taken.text(previous).trim();

			last = taken.remove(previous);
			options.setValue(last);

			return { text, userEdited: taken.userEdited };
		},

		write
	};
}
