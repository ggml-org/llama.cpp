/**
 * Sets the `dark` class from the saved mode-watcher mode, or the system preference
 * when there is none. Docs pages render no story, so preview.ts calls this too.
 */
export function applyColorScheme() {
	const mode = localStorage.getItem('mode-watcher-mode') || 'system';
	const isDark =
		mode === 'dark' ||
		(mode === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);

	document.documentElement.classList.toggle('dark', isDark);
}
