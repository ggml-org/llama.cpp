<script lang="ts">
	import { highlightMatch } from '$lib/utils';

	interface Props {
		text: string;
		query: string;
		matchClass?: string;
	}

	let {
		matchClass = 'rounded bg-yellow-200/60 px-0.5 text-foreground dark:bg-yellow-500/30',
		query,
		text
	}: Props = $props();

	let segments = $derived(highlightMatch(text, query));
</script>

{#each segments as seg, i (i)}
	{#if seg.match}
		<mark class={matchClass}>{seg.text}</mark>
	{:else}
		{seg.text}
	{/if}
{/each}
