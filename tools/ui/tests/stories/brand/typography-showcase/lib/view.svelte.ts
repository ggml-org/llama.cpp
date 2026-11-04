// What the specimen page is showing, shared by the page, its template and the controls. Components take these as
// props, so they render the same anywhere else (a Storybook story, another page) with values of their own.
import { specimen } from './specimen';

export const view = $state({
	comparing: false,
	guides: false,
	size: 64,
	tracking: 0,
	variant: specimen.variants[0], // opens on the first version (Display)
	weight: specimen.styles[0].weight
});

/** The named weight nearest the live one: live text follows any weight, the glyph grid shows named weights. */
export const nearest = (weight: number) =>
	specimen.styles.reduce((a, s) =>
		Math.abs(s.weight - weight) < Math.abs(a.weight - weight) ? s : a
	).weight;

export const variantNotes: Record<string, string> = {
	Display: 'Banners, headlines and large type. Switch to UI at the top (or press V).',
	UI: 'Controls, small interface text and running text. Turn on line boxes in the bar to see the alignment.'
};
