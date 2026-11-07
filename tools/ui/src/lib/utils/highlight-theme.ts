import { UI_DATA_ATTRS } from '$lib/constants';
import { BooleanString } from '$lib/enums';
import githubLightCss from 'highlight.js/styles/github.css?inline';
import githubDarkCss from 'highlight.js/styles/github-dark.css?inline';

/**
 * Loads the highlight.js theme for the given color mode into one shared style
 * element, so every highlighted code block on the page uses the same copy.
 */
export function loadHighlightTheme(isDark: boolean): void {
	if (typeof document === 'undefined') return;

	document
		.querySelectorAll(`style[${UI_DATA_ATTRS.HIGHLIGHT_THEME_PREVIEW}]`)
		.forEach((style) => style.remove());

	const style = document.createElement('style');

	style.setAttribute(UI_DATA_ATTRS.HIGHLIGHT_THEME_PREVIEW, BooleanString.TRUE);
	style.textContent = isDark ? githubDarkCss : githubLightCss;

	document.head.appendChild(style);
}
