<script lang="ts">
	import { base } from '$app/paths';
	import { browser } from '$app/environment';
	import type { Snippet } from 'svelte';

	let { children }: { children?: Snippet } = $props();

	if (browser && 'serviceWorker' in navigator) {
		navigator.serviceWorker.register(`${base}/sw.js`).catch((err) => {
			console.warn('SW registration failed:', err);
		});
	}
</script>

<svelte:head>
	<!-- PWA Web App Manifest -->
	<link rel="manifest" href="{base}/manifest.webmanifest" />

	<!-- Theme Color (matches app background for seamless browser chrome) -->
	<meta name="theme-color" content="#ffffff" media="(prefers-color-scheme: light)" />
	<meta name="theme-color" content="#1b1f20" media="(prefers-color-scheme: dark)" />

	<!-- App Description -->
	<meta name="description" content="A local AI chat interface powered by llama.cpp" />

	<!-- Apple Mobile Web App -->
	<meta name="apple-mobile-web-app-capable" content="yes" />
	<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
	<meta name="apple-mobile-web-app-title" content="llama.cpp" />
	<link rel="apple-touch-icon" href="{base}/favicon.svg" />

	<!-- Android / Chrome -->
	<meta name="mobile-web-app-capable" content="yes" />
</svelte:head>

{@render children?.()}
