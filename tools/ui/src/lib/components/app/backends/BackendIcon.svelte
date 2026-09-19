<script lang="ts">
	import BackendPresetIcon from './BackendPresetIcon.svelte';
	import type { Backend } from '$lib/types';
	import { backendFaviconUrl, findBackendPreset } from '$lib/utils';
	import { SvelteSet } from 'svelte/reactivity';

	/** Favicons that failed to load, so a row remount does not ask again. */
	const failedFavicons = new SvelteSet<string>();

	interface Props {
		backend?: Backend;
		class?: string;
	}

	let { backend, class: className = 'h-4 w-4' }: Props = $props();

	let preset = $derived(backend ? findBackendPreset(backend.baseUrl) : undefined);
	let faviconUrl = $derived(preset || !backend ? null : backendFaviconUrl(backend.baseUrl));
</script>

{#if preset}
	<BackendPresetIcon class={className} {preset} />
{:else if faviconUrl && !failedFavicons.has(faviconUrl)}
	<img
		alt=""
		class={['shrink-0 rounded-sm object-contain', className]}
		decoding="async"
		loading="lazy"
		onerror={() => failedFavicons.add(faviconUrl)}
		src={faviconUrl}
	/>
{/if}
