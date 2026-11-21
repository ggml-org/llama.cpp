import { MEDIA_QUERIES } from '#lib/constants/pwa.constants.ts';
import { ColorMode } from '#lib/enums/ui.enums.ts';
import { modeStorageKey } from 'mode-watcher';

/** The class on `<html>` that turns the dark theme on, as mode-watcher sets it. */
export const DARK_CLASS = ColorMode.DARK;

/** Turns the dark theme on or off. */
export function setDarkClass(isDark: boolean) {
	document.documentElement.classList.toggle(DARK_CLASS, isDark);
}

/**
 * Sets the dark class from the saved mode-watcher mode, or the system preference
 * when there is none. Docs pages render no story, so preview.ts calls this too.
 */
export function applyColorScheme() {
	const mode = localStorage.getItem(modeStorageKey.current) || ColorMode.SYSTEM;
	const isDark =
		mode === ColorMode.DARK ||
		(mode === ColorMode.SYSTEM && window.matchMedia(MEDIA_QUERIES.PREFERS_DARK).matches);

	setDarkClass(isDark);
}
