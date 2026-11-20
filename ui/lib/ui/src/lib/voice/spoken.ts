import {
	VOICE_SPEECH_PEAK_FLOOR,
	VOICE_WHITESPACE_PATTERN,
	VOICE_WORD_PATTERN
} from '../constants/voice.constants.js';

/**
 * How many characters of `text` have been spoken `elapsedMs` into a clip, estimated
 * from its waveform: the text is spread over the parts with speech, so pauses move
 * nothing on, and a word counts as spoken from its start. A transcript without word
 * timings can only be followed this way; it is close, not exact.
 */
export function spokenLength(
	text: string,
	peaks: readonly number[],
	durationMs: number,
	elapsedMs: number
): number {
	if (!(durationMs > 0) || elapsedMs >= durationMs) return text.length;
	if (!(elapsedMs > 0)) return 0;

	const fraction = speechFraction(peaks, elapsedMs / durationMs);

	return wordEnd(text, Math.floor(fraction * text.length));
}

/** The share of the clip's speech that lies before `position` (0..1 of its length). */
export function speechFraction(peaks: readonly number[], position: number): number {
	const loudest = Math.max(0, ...peaks.filter(Number.isFinite));

	if (peaks.length === 0 || loudest <= 0) return clamp(position);

	const weights = peaks.map((peak) =>
		Number.isFinite(peak) && peak > loudest * VOICE_SPEECH_PEAK_FLOOR ? 1 : 0
	);
	const total = weights.reduce<number>((sum, weight) => sum + weight, 0);

	if (total === 0) return clamp(position);

	const at = clamp(position) * weights.length;
	const whole = Math.floor(at);
	let before = 0;

	for (let i = 0; i < whole; i++) before += weights[i];

	if (whole < weights.length) before += weights[whole] * (at - whole);

	return before / total;
}

/** The end of the word `index` falls in, so a word is lit as a whole. */
function wordEnd(text: string, index: number): number {
	if (index <= 0) return 0;
	if (index >= text.length) return text.length;

	const next = text.slice(index).search(VOICE_WHITESPACE_PATTERN);

	return next === -1 ? text.length : index + next;
}

function clamp(value: number): number {
	return Math.min(Math.max(value, 0), 1);
}

/** A text cut into words, each with the whitespace after it and where it starts. */
export function spokenWords(text: string): { start: number; word: string }[] {
	// matchAll works on a copy of the pattern, so the shared one keeps no lastIndex
	return [...text.matchAll(VOICE_WORD_PATTERN)].map((match) => ({
		start: match.index,
		word: match[0]
	}));
}
