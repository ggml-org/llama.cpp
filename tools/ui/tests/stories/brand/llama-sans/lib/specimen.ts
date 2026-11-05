// The specimen's data, exported from the fonts by build_specimen.py (Display comes in via ../../fonts/fonts.css, UI is the design system's own).
import raw from './data/specimen.json';

export interface Cell {
	ch: string;
	adv: number;
	dx: number;
	d: string;
	prev?: string; // the earlier build's outline, when it differs
}

export interface Specimen {
	glyphCount: number;
	metrics: { upm: number; cap: number; x: number; asc: number; dsc: number };
	axis: { min: number; max: number };
	styles: { weight: number; name: string; stem: number }[];
	waterfall: { weight: number; name: string }[];
	variants: string[];
	compare: { labels: string[] } | null;
	grid: {
		variant: string;
		width: number;
		height: number;
		guides: { base: number; x: number; cap: number; asc: number; dsc: number };
		groups: { title: string; chars: string[] }[];
		cells: Record<string, Cell[]>;
	};
}

export const specimen = raw as unknown as Specimen;

/** each version's place on the optical size axis: Llama Sans is one font, and UI and Display are its two ends */
export const opsz: Record<string, number> = { Display: 40, UI: 16 };

export const fallback = '"Helvetica Neue", Arial, sans-serif';
