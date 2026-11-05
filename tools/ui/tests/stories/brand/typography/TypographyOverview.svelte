<!--
	The Typography overview: Llama Sans at its two optical sizes, UI (opsz 16) and Display (opsz 40), and Llama Mono on
	one page, with a switcher pinned at the top. Each Sans tab pins its optical size, so everything in it, large or
	small, shows that end of the axis. Each version gets the same layout, after rsms.me/inter and github.com/mona-sans: a hero set in it, then its own
	part (SansUI, SansDisplay, MonoOverview). It sits in OverviewFrame, which brings the specimen's colours and fonts.
-->
<script lang="ts">
	import MonoOverview from '../llama-mono/MonoOverview.svelte';
	import { specimen } from '../llama-sans/lib/specimen';
	import Hero from '../llama-sans/overview/Hero.svelte';
	import SansDisplay from './SansDisplay.svelte';
	import SansUI from './SansUI.svelte';

	type Variant = 'display' | 'mono' | 'ui';

	// each tab's optical size, pinned (Llama Mono has none)
	const opsz: Record<Variant, string> = { display: "'opsz' 40", mono: 'normal', ui: "'opsz' 16" };

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
			face: 'var(--ui)',
			family: 'Llama Sans',
			id: 'ui',
			kicker: 'Optical size 16 · for interfaces and running text',
			name: 'UI',
			tagline: [
				'Spaced for small sizes.',
				'Centred on the caps.',
				`Every weight from ${axis.min} to ${axis.max}.`
			],
			title: ['Llama Sans', 'UI']
		},
		{
			face: 'var(--headline)',
			family: 'Llama Sans',
			id: 'display',
			kicker: 'Optical size 40 · for headlines and large type',
			name: 'Display',
			tagline: ['The same font as UI.', 'Set tight for large sizes.'],
			title: ['Llama Sans', 'Display']
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

	let selected = $state<Variant>('ui');
	const current = $derived(variants.find((v) => v.id === selected)!);

	function select(id: Variant) {
		if (id === selected) return;

		selected = id;
		// a new version starts at its hero, wherever the switcher was used
		window.scrollTo({ top: 0 });
	}

	// left and right arrows move between the tabs, as in any tablist
	function onkeydown(event: KeyboardEvent) {
		const step = { ArrowLeft: -1, ArrowRight: 1 }[event.key];

		if (!step) return;

		const i =
			(variants.findIndex((v) => v.id === selected) + step + variants.length) % variants.length;

		select(variants[i].id);
		(event.currentTarget as HTMLElement)
			.querySelector<HTMLElement>(`#tab-${variants[i].id}`)
			?.focus();
	}
</script>

<div class="page">
	<nav class="switcher">
		<div aria-label="Typeface" class="tabs" {onkeydown} role="tablist" tabindex="-1">
			{#each variants as v (v.id)}
				<button
					style:--face={v.face}
					style:--opsz-pin={opsz[v.id]}
					aria-controls="panel"
					aria-selected={v.id === selected}
					class="tab"
					id="tab-{v.id}"
					onclick={() => select(v.id)}
					role="tab"
					tabindex={v.id === selected ? 0 : -1}
					type="button"
				>
					<span class="family">{v.family}</span>

					<span class="name">{v.name}</span>
				</button>
			{/each}
		</div>
	</nav>

	<div
		style:--opsz-pin={opsz[selected]}
		aria-labelledby="tab-{selected}"
		class="panel"
		id="panel"
		role="tabpanel"
	>
		{#key selected}
			<Hero
				face={current.face}
				kicker={current.kicker}
				tagline={current.tagline}
				title={current.title}
			/>

			{#if selected === 'ui'}
				<SansUI />
			{:else if selected === 'display'}
				<SansDisplay />
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
	.tabs {
		display: flex;
		gap: 4px;
		padding: 4px;
		border-radius: 999px;
		background: var(--sheet);
	}
	.tab {
		display: inline-flex;
		align-items: baseline;
		gap: 6px;
		padding: 8px 16px;
		border: 0;
		border-radius: 999px;
		background: transparent;
		color: var(--muted);
		font: 600 15px / 1.25 var(--face);
		cursor: pointer;
		transition:
			background-color 150ms ease,
			color 150ms ease;
	}
	.tab:hover {
		color: var(--ink);
	}
	.tab[aria-selected='true'] {
		background: var(--paper);
		color: var(--ink);
		box-shadow:
			0 1px 2px rgb(0 0 0 / 0.08),
			0 0 0 1px var(--rule);
	}
	/* in dark the paper is darker than the track, so the selected tab rises to a lighter grey instead */
	:global(.dark) .tab[aria-selected='true'] {
		background: color-mix(in oklab, var(--ink) 14%, var(--sheet));
		box-shadow: none;
	}
	.tab:focus-visible {
		outline: 2px solid var(--mark);
		outline-offset: 2px;
	}
	.family {
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
