<script lang="ts">
	import { Plug } from '@lucide/svelte';
	import type { BackendPreset } from '$lib/types';
	import { mode } from 'mode-watcher';

	interface Props {
		class?: string;
		preset?: BackendPreset;
	}

	let { class: className = 'h-4 w-4', preset }: Props = $props();

	let iconUrl = $derived.by(() => {
		if (!preset) return null;

		return (
			(mode.current === 'dark'
				? (preset.iconUrlDark ?? preset.iconUrl)
				: (preset.iconUrlLight ?? preset.iconUrl)) ?? null
		);
	});
</script>

{#if iconUrl}
	<img
		alt=""
		class={['shrink-0 rounded-sm object-contain', className]}
		decoding="async"
		loading="lazy"
		src={iconUrl}
	/>
{:else}
	<Plug class={className} />
{/if}
