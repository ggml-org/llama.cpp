import { UI_DATA_ATTRS } from '$lib/constants';
import { BooleanString } from '$lib/enums';
import githubLightCss from 'highlight.js/styles/github.css?inline';
import githubDarkCss from 'highlight.js/styles/github-dark.css?inline';

/**
 * Loads the highlight.js theme for the given color mode into one shared style
 * element, so every highlighted code block on the page uses the same copy. Every
 * message and editor asks for it as it mounts: when the theme is already there,
 * nothing changes, as swapping the stylesheet restyles the whole page.
 */
export function loadHighlightTheme(isDark: boolean): void {
	if (typeof document === 'undefined') return;

	const css = isDark ? githubDarkCss : githubLightCss;
	const styles = document.querySelectorAll<HTMLStyleElement>(
		`style[${UI_DATA_ATTRS.HIGHLIGHT_THEME_PREVIEW}]`
	);

	if (styles.length === 1 && styles[0].textContent === css) return;

	styles.forEach((style) => style.remove());

	const style = document.createElement('style');

	style.setAttribute(UI_DATA_ATTRS.HIGHLIGHT_THEME_PREVIEW, BooleanString.TRUE);
	style.textContent = css;

	document.head.appendChild(style);
}
