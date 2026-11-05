// Colour for communications (posters, the site, social): a neutral base of the design system's greys, and the poudre
// palette, which adds a powdered Swiss blue, sky, navy, warm stone, beige and a terracotta accent on top of it. The UI itself
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

/** The poudre palette: the classic Swiss specimen boards' hues, powdered towards the beige and blues of the Llama Sans
 *  Display In use story and pulled to the design system's status hues: a powdered Swiss blue as the dark field, warm
 *  stone greys, beige, and a terracotta accent (between the danger red and clay; the brand orange stays with the
 *  logotype), with sky, teal, saffron and a poudre pink in small doses. */
const SWISS = 'linear-gradient(160deg, #517dae 0%, #4671a5 55%, #3d6293 100%)';
const GREY = 'linear-gradient(160deg, #a69d93 0%, #bdb6ad 55%, #d9d3cc 100%)';

export const poudrePalette: Palette = {
	name: 'Poudre',
	note: 'Powdered: a soft Swiss blue field, warm stone greys, beige and a terracotta accent, with sky, teal, saffron and poudre pink in small doses',
	roles: {
		...base,
		accent: '#c97152',
		accentInk: '#ffffff',
		deep: SWISS,
		deepTone: '#3d6293',
		ink: '#2b2620',
		soft: GREY,
		softInk: '#2b2620',
		surface: '#faf7f3',
		warm: '#efe9df'
	},
	swatches: [
		{ name: 'Swiss blue', value: '#4671a5' },
		{ name: 'Sky', value: '#9bbfdf' },
		{ name: 'Navy', value: NAVY },
		{ name: 'Stone', value: GREY },
		{ name: 'Beige', value: '#efe9df' },
		{ name: 'Terracotta · accent', value: '#c97152' },
		{ name: 'Saffron', value: '#e1b671' },
		{ name: 'Teal', value: '#70a49e' },
		{ name: 'Poudre', value: '#edc6c6' },
		{ name: 'Sage', value: '#daddcc' }
	]
};
