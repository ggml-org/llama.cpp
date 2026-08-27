<script lang="ts">
	import { APP_NAME, APPLE_META_TAGS, MEDIA_QUERIES, THEME_COLORS } from '$lib/constants';

	let { appName = APP_NAME } = $props();
</script>

<svelte:head>
	<!-- Theme color for light/dark modes -->
	<meta content={THEME_COLORS.LIGHT} media={MEDIA_QUERIES.PREFERS_LIGHT} name="theme-color" />

	<meta content={THEME_COLORS.DARK} media={MEDIA_QUERIES.PREFERS_DARK} name="theme-color" />

	<!-- Apple mobile web app meta tags -->
	<meta
		content={APPLE_META_TAGS.MOBILE_WEB_APP_CAPABLE.content}
		name={APPLE_META_TAGS.MOBILE_WEB_APP_CAPABLE.name}
	/>

	<meta
		content={APPLE_META_TAGS.STATUS_BAR_STYLE.content}
		name={APPLE_META_TAGS.STATUS_BAR_STYLE.name}
	/>

	<meta content={appName} name={APPLE_META_TAGS.MOBILE_WEB_APP_TITLE.name} />
</svelte:head>
