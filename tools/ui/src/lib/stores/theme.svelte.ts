import { browser } from '$app/environment';
import { MEDIA_QUERIES } from '$lib/constants';

export const theme = $state({
	isSystemDark: browser && window.matchMedia(MEDIA_QUERIES.PREFERS_DARK).matches
});

if (browser) {
	const mql = window.matchMedia(MEDIA_QUERIES.PREFERS_DARK);

	mql.addEventListener('change', (e) => {
		theme.isSystemDark = e.matches;
	});
}
