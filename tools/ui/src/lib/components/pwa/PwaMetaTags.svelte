<script lang="ts">
	import { APPLE_META_TAGS, MEDIA_QUERIES, THEME_COLORS } from '$lib/constants/pwa';
	import { APP_NAME } from '$lib/constants';

	let { appName = APP_NAME } = $props();
</script>

<svelte:head>
	<!-- Theme color for light/dark modes -->
	<meta name="theme-color" content={THEME_COLORS.LIGHT} media={MEDIA_QUERIES.PREFERS_LIGHT} />
	<meta name="theme-color" content={THEME_COLORS.DARK} media={MEDIA_QUERIES.PREFERS_DARK} />

	<!-- Apple mobile web app meta tags -->
	<meta
		name={APPLE_META_TAGS.MOBILE_WEB_APP_CAPABLE.name}
		content={APPLE_META_TAGS.MOBILE_WEB_APP_CAPABLE.content}
	/>
	<meta
		name={APPLE_META_TAGS.STATUS_BAR_STYLE.name}
		content={APPLE_META_TAGS.STATUS_BAR_STYLE.content}
	/>
	<meta name={APPLE_META_TAGS.MOBILE_WEB_APP_TITLE.name} content={appName} />
</svelte:head>
