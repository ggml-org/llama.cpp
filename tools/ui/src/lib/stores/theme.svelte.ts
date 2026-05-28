import { browser } from '$app/environment';

export const theme = $state({
	isSystemDark: browser && window.matchMedia('(prefers-color-scheme: dark)').matches
});

if (browser) {
	const mql = window.matchMedia('(prefers-color-scheme: dark)');

	mql.addEventListener('change', (e) => {
		theme.isSystemDark = e.matches;
	});
}
