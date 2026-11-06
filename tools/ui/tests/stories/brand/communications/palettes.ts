// Colour for communications (posters, the site, social): a neutral base of the design system's greys, and the Llama
// palette (the day sky, the night sky, sand and cinder) on top of it. The UI itself
// stays neutral and uses colour only for status (in the tokens; see DESIGN.md). Cinder, the accent orange, is also the .cpp of the
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
