// Colour for communications (posters, the site, social): a neutral base of the design system's greys, and the Haas
// palette, which adds a Swiss blue, sky, navy, cool grey, cream and a vermilion accent on top of it. The UI itself
// stays neutral and uses colour only for status (in the tokens; see DESIGN.md). The brand orange (#f65e00) stays with the
// llama.cpp logotype.

export interface Palette {
	name: string;
	note: string;
	/** the swatches to show, in order */
	swatches: { name: string; value: string }[];
	/** the roles, as CSS custom properties for the posters */
	roles: {
		accent: string; // the one accent: a mark, a highlight, one sticker (never in UI chrome)
		accentInk: string; // text on the accent
		deep: string; // the dark field (navy)
		deepInk: string;
		deepTone: string; // a flat colour from the dark field, for small shapes
		field: string; // a quiet background (sage)
		ink: string; // text on the light surfaces
		line: string;
		soft: string; // the light field (mist)
		softInk: string;
		surface: string; // the plain light surface (neutral)
		warm: string; // a warm light surface (sand)
	};
}

const NAVY =
	'radial-gradient(ellipse at 15% 10%, #22385c 0%, transparent 55%), radial-gradient(ellipse at 90% 95%, #1e3752 0%, transparent 50%), linear-gradient(160deg, #152238 0%, #172a42 55%, #1a3049 100%)';
const MIST =
	'radial-gradient(ellipse at 15% 10%, #dcecfb 0%, transparent 55%), radial-gradient(ellipse at 90% 95%, #e8f2fc 0%, transparent 50%), linear-gradient(160deg, #c4ddf6 0%, #d0e4f7 55%, #dfecf9 100%)';
const base = {
	deep: NAVY,
	deepInk: '#ffffff',
	deepTone: '#1d3250',
	field: '#d8dfd1',
	ink: '#0f1f33',
	line: 'rgb(15 31 51 / 0.14)',
	soft: MIST,
	softInk: '#0f1f33',
	surface: '#fafafa',
	warm: '#efe9df'
};

/** The base on its own: the design system's greys only, every role neutral. */
export const basePalette: Palette = {
	name: 'Base',
	note: "The design system's greys alone: every surface, field and accent neutral",
	roles: {
		accent: '#404040',
		accentInk: '#fafafa',
		deep: 'linear-gradient(160deg, #1c1c1c 0%, #171717 55%, #141414 100%)',
		deepInk: '#fafafa',
		deepTone: '#262626',
		field: '#e5e5e5',
		ink: '#0a0a0a',
		line: 'rgb(0 0 0 / 0.12)',
		soft: 'linear-gradient(160deg, #e5e5e5 0%, #ebebeb 55%, #f0f0f0 100%)',
		softInk: '#0a0a0a',
		surface: '#fafafa',
		warm: '#f0f0f0'
	},
	swatches: [
		{ name: 'Neutral 50', value: '#fafafa' },
		{ name: 'Neutral 100', value: '#f5f5f5' },
		{ name: 'Neutral 200', value: '#e5e5e5' },
		{ name: 'Neutral 500', value: '#737373' },
		{ name: 'Neutral 700', value: '#404040' },
		{ name: 'Neutral 900', value: '#171717' },
		{ name: 'Neutral 950', value: '#0a0a0a' }
	]
};

/** After the classic Swiss specimen boards: the base made livelier, with a Swiss blue as the dark field, a deeper
 *  cool grey, cream, and a vermilion accent (redder and calmer than the brand orange, which stays with the logotype);
 *  saturated sky, teal, saffron and pink appear in small doses on the boards. */
const SWISS = 'linear-gradient(160deg, #3474bd 0%, #2d67ab 55%, #275b99 100%)';
const GREY = 'linear-gradient(160deg, #8a99a4 0%, #b0bbc2 55%, #d3d9dd 100%)';

export const haasPalette: Palette = {
	name: 'Haas',
	note: 'Livelier: a Swiss blue field, a deeper cool grey, cream and a vermilion accent, with sky, teal, saffron and pink in small doses',
	roles: {
		...base,
		accent: '#d9553b',
		accentInk: '#ffffff',
		deep: SWISS,
		deepTone: '#245a9a',
		soft: GREY,
		softInk: '#0f1f33',
		surface: '#fbf9f6',
		warm: '#f2e9dc'
	},
	swatches: [
		{ name: 'Swiss blue', value: '#2f6db5' },
		{ name: 'Sky', value: '#7eb6e6' },
		{ name: 'Navy', value: NAVY },
		{ name: 'Cool grey', value: GREY },
		{ name: 'Cream', value: '#f2e9dc' },
		{ name: 'Vermilion · accent', value: '#d9553b' },
		{ name: 'Saffron', value: '#eda634' },
		{ name: 'Teal', value: '#4fa5a0' },
		{ name: 'Pink', value: '#efb8bf' },
		{ name: 'Sage', value: '#d8dfd1' }
	]
};
