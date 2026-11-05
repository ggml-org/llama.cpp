// Colour for communications (posters, the site, social): a neutral base of the design system's greys, and the Llama
// palette, the design system's own blues, green, amber and red with beige as the warm neutral, on top of it. The UI itself
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
	'radial-gradient(ellipse at 15% 10%, color-mix(in oklab, var(--blue-800), black 15%) 0%, transparent 55%), linear-gradient(160deg, color-mix(in oklab, var(--blue-800), black 38%) 0%, color-mix(in oklab, var(--blue-800), black 28%) 100%)';
const MIST =
	'linear-gradient(160deg, color-mix(in oklab, var(--blue-400), white 62%) 0%, color-mix(in oklab, var(--blue-400), white 72%) 100%)';
const base = {
	deep: NAVY,
	deepInk: 'var(--neutral-0)',
	deepTone: 'color-mix(in oklab, var(--blue-800), black 28%)',
	field: 'var(--beige-200)',
	ink: 'var(--neutral-950)',
	line: 'rgb(0 0 0 / 0.12)',
	soft: MIST,
	softInk: 'var(--neutral-950)',
	surface: 'var(--neutral-50)',
	warm: 'var(--beige-100)'
};

/** The base on its own: the design system's greys only, every role neutral. */
export const basePalette: Palette = {
	name: 'Base',
	note: "The design system's greys alone: every surface, field and accent neutral",
	roles: {
		accent: 'var(--neutral-700)',
		accentInk: 'var(--neutral-50)',
		deep: 'linear-gradient(160deg, var(--neutral-900) 0%, var(--neutral-925) 55%, var(--neutral-950) 100%)',
		deepInk: 'var(--neutral-50)',
		deepTone: 'var(--neutral-800)',
		field: 'var(--neutral-200)',
		ink: 'var(--neutral-950)',
		line: 'rgb(0 0 0 / 0.12)',
		soft: 'linear-gradient(160deg, var(--neutral-200) 0%, var(--neutral-150) 100%)',
		softInk: 'var(--neutral-950)',
		surface: 'var(--neutral-50)',
		warm: 'var(--neutral-150)'
	},
	swatches: [
		{ name: 'Neutral 50', value: 'var(--neutral-50)' },
		{ name: 'Neutral 100', value: 'var(--neutral-100)' },
		{ name: 'Neutral 200', value: 'var(--neutral-200)' },
		{ name: 'Neutral 500', value: 'var(--neutral-500)' },
		{ name: 'Neutral 700', value: 'var(--neutral-700)' },
		{ name: 'Neutral 900', value: 'var(--neutral-900)' },
		{ name: 'Neutral 950', value: 'var(--neutral-950)' }
	]
};

/** The design system's palette for communications: neutral fields (the greys and beige) as a rule, and its blues,
 *  red, green and amber where colour serves a purpose: shapes, numerals and accents (the brand orange stays with the
 *  logotype). Every colour is a palette token or a mix within one hue. */
const GREY =
	'linear-gradient(160deg, var(--neutral-400) 0%, var(--neutral-300) 55%, var(--neutral-200) 100%)';

export const llamaPalette: Palette = {
	name: 'Llama',
	note: "The design system's own colours on neutral fields: greys and beige behind, blues, red, green and amber for shapes and accents",
	roles: {
		...base,
		accent: 'var(--red-550)',
		accentInk: '#ffffff',
		deep: 'linear-gradient(160deg, var(--neutral-800) 0%, var(--neutral-900) 100%)',
		deepTone: 'var(--neutral-800)',
		ink: 'var(--beige-900)',
		soft: GREY,
		softInk: 'var(--beige-900)',
		surface: 'var(--beige-50)',
		warm: 'var(--beige-100)'
	},
	swatches: [
		{ name: 'Blue 550 · Swiss', value: 'var(--blue-550)' },
		{ name: 'Blue 400 · sky', value: 'var(--blue-400)' },
		{ name: 'Navy', value: NAVY },
		{ name: 'Greys', value: GREY },
		{ name: 'Beige 100', value: 'var(--beige-100)' },
		{ name: 'Red 550 · accent', value: 'var(--red-550)' },
		{ name: 'Amber 400', value: 'var(--amber-400)' },
		{ name: 'Green 550', value: 'var(--green-550)' },
		{ name: 'Light red', value: 'color-mix(in oklab, var(--red-400), white 62%)' },
		{ name: 'Beige 50', value: 'var(--beige-50)' }
	]
};
