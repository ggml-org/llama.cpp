<!--
	A post's top row: the brand (the Llama lockup, llama.cpp's in Llama Mono, or the icon alone before a Mono
	kicker), an optional pill with a cinder dot, and the URL in Mono on the right.
-->
<script lang="ts">
	import { ICON, LOCKUP } from './post';

	interface Props {
		brand: 'icon' | 'Llama' | 'llama.cpp';
		kicker?: string;
		pill?: string;
		/** the lockup's size in units; the kicker, pill and URL follow it */
		size?: number;
		url?: string;
	}

	let { brand, kicker, pill, size = 0.8, url }: Props = $props();
</script>

<header style:--s="calc(var(--u) * {size})" class="row">
	{#if brand === 'icon'}
		<span aria-label="Llama" class="icon" role="img">{ICON}</span>
	{:else if brand === 'Llama'}
		<span class="lockup">{LOCKUP}Llama</span>
	{:else}
		<span class="lockup mono">{LOCKUP}llama<span class="cpp">.cpp</span></span>
	{/if}

	{#if kicker}<span class="kicker">{kicker}</span>{/if}

	{#if pill}
		{#if brand !== 'icon'}<i class="rule"></i>{/if}

		<span class="pill"><i class="dot"></i>{pill}</span>
	{/if}

	{#if url}<span class="url">{url}</span>{/if}
</header>

<style>
	.row {
		display: flex;
		align-items: center;
		gap: calc(var(--s) * 0.5);
		white-space: nowrap;
	}
	.lockup {
		font: 700 var(--s) / 1 var(--sans);
		letter-spacing: -0.01em;
	}
	/* Llama Mono's U+E002 is drawn 1 / 0.9 times Llama Sans's: at 90% its icon matches Llama's */
	.lockup.mono {
		font-family: var(--mono);
		font-size: calc(var(--s) * 0.9);
		letter-spacing: 0;
	}
	.cpp {
		color: var(--art-cinder);
	}
	.icon {
		font: 700 calc(var(--s) * 1.4) / 1 var(--sans);
	}
	/* the kicker a little smaller and less tracked than the URL, since it runs longer */
	.kicker {
		margin-left: calc(var(--s) * 0.2);
		color: var(--p-muted);
		font: 400 calc(var(--s) * 0.62) / 1 var(--mono);
		letter-spacing: 0.02em;
	}
	.url {
		margin-left: auto;
		color: var(--p-ink);
		font: 400 calc(var(--s) * 0.76) / 1 var(--mono);
	}
	.rule {
		align-self: stretch;
		width: 2px;
		margin-inline: calc(var(--s) * 0.05);
		background: var(--p-line);
	}
	.pill {
		display: inline-flex;
		align-items: center;
		gap: calc(var(--s) * 0.28);
		padding: calc(var(--s) * 0.24) calc(var(--s) * 0.42);
		border: 2px solid var(--p-line);
		border-radius: 999px;
		color: var(--p-ink);
		font: 500 calc(var(--s) * 0.68) / 1 var(--sans);
	}
	.dot {
		width: calc(var(--s) * 0.22);
		height: calc(var(--s) * 0.22);
		border-radius: 50%;
		background: var(--art-cinder);
	}
</style>
