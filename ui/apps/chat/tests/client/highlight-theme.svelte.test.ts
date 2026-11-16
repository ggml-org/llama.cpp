// Every message and editor loads the highlight theme as it mounts; swapping the style
// element restyles the whole page, so it only happens when the mode changes.

import { UI_DATA_ATTRS } from '$lib/constants';
import { loadHighlightTheme } from '$lib/utils';
import { describe, expect, it } from 'vitest';

const themes = () => document.querySelectorAll(`style[${UI_DATA_ATTRS.HIGHLIGHT_THEME_PREVIEW}]`);

describe('loadHighlightTheme', () => {
	it('keeps the style element while the mode stays and swaps it on a change', () => {
		loadHighlightTheme(false);

		const light = themes()[0];

		loadHighlightTheme(false);

		expect(themes()).toHaveLength(1);
		expect(themes()[0]).toBe(light);

		loadHighlightTheme(true);

		expect(themes()).toHaveLength(1);
		expect(themes()[0]).not.toBe(light);
	});
});
