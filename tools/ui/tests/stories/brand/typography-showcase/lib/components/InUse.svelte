<!-- The version on show at work: Display's headlines, or UI's controls and then its running text. -->
<script lang="ts">
	import ShowcaseDisplay from './ShowcaseDisplay.svelte';
	import ShowcaseProse from './ShowcaseProse.svelte';
	import ShowcaseUI from './ShowcaseUI.svelte';

	let {
		glyphCount,
		guides = false,
		variant
	}: { variant: string; glyphCount: number; guides?: boolean } = $props();
</script>

<section aria-label="In use" class="in-use">
	{#if variant === 'Display'}
		<ShowcaseDisplay />
	{:else}
		<ShowcaseUI {glyphCount} {guides} />

		<ShowcaseProse />
	{/if}
</section>

<style>
	.in-use {
		display: grid;
		gap: 56px;
	}
</style>
