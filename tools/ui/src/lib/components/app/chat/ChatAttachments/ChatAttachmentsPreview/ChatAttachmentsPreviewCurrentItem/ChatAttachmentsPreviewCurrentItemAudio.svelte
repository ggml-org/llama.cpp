<script lang="ts">
	import { Music } from '@lucide/svelte';

	interface Props {
		currentItem: { name?: string } | null;
		audioSrc: string | null;
	}

	let { currentItem, audioSrc }: Props = $props();
</script>

<div class="flex flex-1 items-center justify-center p-8">
	<div class="w-full max-w-md text-center">
		<Music class="mx-auto mb-4 h-16 w-16 text-white/50" />

		{#if audioSrc}
			<audio controls class="mb-4 w-full" src={audioSrc}>
				Your browser does not support the audio element.
			</audio>
		{:else}
			<p class="mb-4 text-white/70">Audio preview not available</p>
		{/if}

		<p class="text-sm text-white/50">{currentItem?.name || 'Audio'}</p>
	</div>
</div>
