// The lockup proposal's geometry: the icon-to-wordmark proportion in even steps from today's lockup (ggml-org/
// llama.brand, build_lockups.py: the icon 1.91× the cap height, 0.81 cap heights before the first letter) to the icon
// as a glyph of Llama Sans (U+E000: 1× the cap height, then its sidebearing, a space and the first letter's sidebearing).

export const CAP = 710; // Llama Sans's cap height, in font units (the wordmark is set at 1000)
export const STEPS = 6;

const FROM = { gap: 221.5 / 272.6, icon: 520 / 272.6 };
const TO = { gap: (49 + 213 + 69) / CAP, icon: 1 };

export interface Step {
	gap: number;
	glyph: boolean;
	icon: number;
	label: string;
}

export function step(i: number): Step {
	const t = i / (STEPS - 1);

	return {
		gap: FROM.gap + (TO.gap - FROM.gap) * t,
		glyph: i === STEPS - 1,
		icon: FROM.icon + (TO.icon - FROM.icon) * t,
		label: i === STEPS - 1 ? 'The glyph' : `Step ${i}`
	};
}

// each wordmark's first letter's left sidebearing, in font units, so the gap is measured to its ink
export const FIRST_LSB: Record<string, number> = { Llama: 69, 'llama.cpp': 59 };

/** A step between the even ones, by its icon size (× the cap height); the gap follows the same line. */
export function stepAt(icon: number, label: string): Step {
	const t = (FROM.icon - icon) / (FROM.icon - TO.icon);

	return { gap: FROM.gap + (TO.gap - FROM.gap) * t, glyph: false, icon, label };
}
