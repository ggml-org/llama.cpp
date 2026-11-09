<!--
	A font's Typography overview, one for Llama Sans and one for Llama Mono, each first under its own font. Both get the
	same layout, after rsms.me/inter and github.com/mona-sans: a hero set in it, then its own part (SansOverview, with a
	part for each end of its optical size axis, and MonoOverview). It sits in OverviewFrame, which brings the
	specimen's colours and fonts.
-->
<script lang="ts">
	import MonoOverview from '../llama-mono/MonoOverview.svelte';
	import { specimen } from '../llama-sans/lib/specimen';
	import Hero from '../llama-sans/overview/Hero.svelte';
	import SansOverview from './SansOverview.svelte';

	interface Props {
		variant: 'mono' | 'sans';
	}

	let { variant }: Props = $props();

	const { axis } = specimen;

	const variants = {
		mono: {
			face: 'var(--code)',
			kicker: 'For code and numbers',
			name: 'Mono',
			tagline: ['Every character, the same cell.', 'Llama Sans’s letters, set for code.'],
			title: ['Llama', 'Mono']
		},
		sans: {
			face: 'var(--headline)',
			kicker: 'For interfaces, reading and headlines',
			name: 'Sans',
			tagline: [
				'One font for every size.',
				'Spaced for text at 16, tight for headlines at 40.',
				`Every weight from ${axis.min} to ${axis.max}.`
			],
			title: ['Llama', 'Sans']
		}
	};

	const current = $derived(variants[variant]);
</script>

<div aria-label="Llama {current.name}" class="panel" role="region">
	<Hero
		face={current.face}
		kicker={current.kicker}
		tagline={current.tagline}
		title={current.title}
	/>

	{#if variant === 'sans'}
		<SansOverview />
	{:else}
		<MonoOverview />
	{/if}
</div>

<style>
	.panel {
		display: grid;
		gap: clamp(64px, 9vw, 128px);
		min-width: 0;
	}
	.panel > :global(*) {
		min-width: 0;
	}
</style>
