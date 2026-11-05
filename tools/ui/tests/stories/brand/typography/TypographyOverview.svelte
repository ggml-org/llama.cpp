<!--
	The Typography overview: Llama Sans and Llama Mono on one page, with the design system's Tabs pinned at the top to
	switch between them. Each font gets the
	same layout, after rsms.me/inter and github.com/mona-sans: a hero set in it, then its own part (SansOverview, with a
	part for each end of its optical size axis, and MonoOverview). It sits in OverviewFrame, which brings the
	specimen's colours and fonts.
-->
<script lang="ts">
	import MonoOverview from '../llama-mono/MonoOverview.svelte';
	import { specimen } from '../llama-sans/lib/specimen';
	import Hero from '../llama-sans/overview/Hero.svelte';
	import SansOverview from './SansOverview.svelte';
	import * as Tabs from '$lib/components/ui/tabs';

	type Variant = 'mono' | 'sans';

	const { axis } = specimen;

	const variants: {
		face: string;
		family: string;
		id: Variant;
		kicker: string;
		name: string;
		tagline: string[];
		title: string[];
	}[] = [
		{
			face: 'var(--headline)',
			family: 'Llama',
			id: 'sans',
			kicker: 'For interfaces, reading and headlines',
			name: 'Sans',
			tagline: [
				'One font for every size.',
				'Spaced for text at 16, tight for headlines at 40.',
				`Every weight from ${axis.min} to ${axis.max}.`
			],
			title: ['Llama', 'Sans']
		},
		{
			face: 'var(--code)',
			family: 'Llama',
			id: 'mono',
			kicker: 'For code and numbers',
			name: 'Mono',
			tagline: ['Every character, the same cell.', 'Llama Sans’s letters, set for code.'],
			title: ['Llama', 'Mono']
		}
	];

	let selected = $state<Variant>('sans');
	const current = $derived(variants.find((v) => v.id === selected)!);
</script>

<div class="page">
	<!-- the design system's Tabs (in .ds, so the specimen's reset leaves them alone); a new font starts at its hero,
	     wherever the switcher was used -->
	<nav class="switcher">
		<Tabs.Root
			bind:value={selected}
			class="ds items-center"
			onValueChange={() => window.scrollTo({ top: 0 })}
		>
			<Tabs.List aria-label="Typeface">
				{#each variants as v (v.id)}
					<Tabs.Trigger style="font-family: {v.face}" value={v.id}>
						<span class="family">{v.family}</span>

						{v.name}
					</Tabs.Trigger>
				{/each}
			</Tabs.List>
		</Tabs.Root>
	</nav>

	<div aria-label="Llama {current.name}" class="panel" role="region">
		{#key selected}
			<Hero
				face={current.face}
				kicker={current.kicker}
				tagline={current.tagline}
				title={current.title}
			/>

			{#if selected === 'sans'}
				<SansOverview />
			{:else}
				<MonoOverview />
			{/if}
		{/key}
	</div>
</div>

<style>
	/* one item in the frame's grid, so the switcher sits right on the hero, with no gap */
	.page {
		display: grid;
		min-width: 0;
	}
	.switcher {
		position: sticky;
		top: 0;
		z-index: 10;
		display: flex;
		justify-content: center;
		padding: 12px 16px;
		background: color-mix(in oklab, var(--paper) 82%, transparent);
		backdrop-filter: blur(12px);
		border-bottom: 1px solid var(--rule);
	}
	.family {
		margin-right: 0.25em;
		font-weight: 400;
	}
	@media (max-width: 480px) {
		.family {
			display: none;
		}
	}
	.panel {
		display: grid;
		gap: clamp(64px, 9vw, 128px);
		min-width: 0;
	}
	.panel > :global(*) {
		min-width: 0;
	}
</style>
