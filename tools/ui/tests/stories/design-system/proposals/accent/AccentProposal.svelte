<!--
	Proposal: one accent orange for the brand, in place of the three in use today (clay #a8704f in the specimen,
	vermilion #d9553b in the overview, the Mono key visuals and art-orange, and the brand orange #f65e00 still defined for
	the logotype). A switcher sets each candidate as all three at once, over the real Display gallery, Llama Mono's key
	visuals and the llama.cpp wordmark, so it can be judged next to the hero blue, navy, beige, white and ink.
-->
<script lang="ts">
	import KeyVisuals from '../../../brand/llama-mono/KeyVisuals.svelte';
	import DisplayGallery from '../../../brand/llama-sans/lib/components/DisplayGallery.svelte';
	import SpecimenFrame from '../../../brand/llama-sans/SpecimenFrame.svelte';
	import * as ToggleGroup from '$lib/components/ui/toggle-group';

	// contrast (WCAG) against neutral-50, beige-100, the hero's light blue, art-navy and the dark page
	const candidates = [
		{
			contrast: { beige: 2.81, dark: 5.72, light: 3.25, navy: 3.61, sky: 2.61 },
			hex: '#e3662f',
			id: 'persimmon',
			label: 'Persimmon',
			note: 'A step from today’s vermilion toward true orange, still strong on white.',
			value: 'oklch(0.655 0.17 42)'
		},
		{
			contrast: { beige: 2.54, dark: 6.34, light: 2.93, navy: 4, sky: 2.35 },
			hex: '#e9712b',
			id: 'mandarin',
			label: 'Mandarin',
			note: 'Halfway between persimmon and tangerine: warm on white, clear on navy.',
			value: 'oklch(0.68 0.168 47)'
		},
		{
			contrast: { beige: 2.29, dark: 7.02, light: 2.65, navy: 4.43, sky: 2.12 },
			hex: '#ee7d29',
			id: 'tangerine',
			label: 'Tangerine',
			note: 'The blue’s complement: vivid, best on navy and ink.',
			value: 'oklch(0.705 0.165 52)'
		}
	];

	let selected = $state('mandarin');
	const accent = $derived(candidates.find((c) => c.id === selected)!);

	const grounds = [
		['light', 'White', '#fafafa'],
		['beige', 'Beige', 'oklch(0.936 0.015 81)'],
		['sky', 'Hero blue', '#d0e4f7'],
		['navy', 'Navy', 'oklch(0.33 0.07 256)'],
		['dark', 'Ink', '#111111']
	] as const;
</script>

<SpecimenFrame variant="Display">
	<div
		style:--art-orange={accent.value}
		style:--brand-orange={accent.value}
		style:--clay={accent.value}
		style:--vermilion={accent.value}
		class="proposal"
	>
		<div class="picker ds">
			<ToggleGroup.Root
				bind:value={() => selected, (v) => v && (selected = v)}
				aria-label="Accent orange"
				type="single"
			>
				{#each candidates as c (c.id)}
					<ToggleGroup.Item value={c.id}>
						<span style:background={c.value} aria-hidden="true" class="dot"></span>{c.label}
					</ToggleGroup.Item>
				{/each}
			</ToggleGroup.Root>

			<p class="note">
				<code>{accent.value}</code> · <code>{accent.hex}</code> · {accent.note}
			</p>
		</div>

		<section aria-label="The accent on each ground" class="grounds">
			{#each grounds as [key, name, bg] (key)}
				<div class:on-dark={key === 'navy' || key === 'dark'} style:background={bg} class="ground">
					<p class="word">llama<span>.cpp</span></p>

					<p class="meta">{name} · {accent.contrast[key].toFixed(2)}:1</p>
				</div>
			{/each}
		</section>

		<DisplayGallery />

		<KeyVisuals />
	</div>
</SpecimenFrame>

<style>
	.proposal {
		--code: 'Llama Mono', ui-monospace, monospace; /* the key visuals are set in Llama Mono */
		display: grid;
		gap: 40px;
	}
	.picker {
		position: sticky;
		top: 0;
		z-index: 5;
		display: grid;
		gap: 10px;
		padding: 12px 0;
		background: color-mix(in oklab, var(--paper) 88%, transparent);
		backdrop-filter: blur(12px);
		border-bottom: 1px solid var(--rule);
	}
	.dot {
		display: inline-block;
		width: 10px;
		height: 10px;
		margin-right: 6px;
		border-radius: 999px;
	}
	.note {
		margin: 0;
		font: 400 13px / 1.4 var(--body);
		color: var(--muted);
	}
	.note code {
		font: 500 12px var(--mono);
		color: var(--ink);
	}
	.grounds {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
		gap: 12px;
	}
	.ground {
		display: grid;
		gap: 24px;
		padding: 24px 20px 16px;
		border-radius: 14px;
		color: #111111;
	}
	.ground.on-dark {
		color: #fafafa;
	}
	.ground > * {
		margin: 0;
	}
	.word {
		font:
			700 32px / 1.1 'Llama Sans',
			var(--body);
		letter-spacing: -0.01em;
	}
	.word span {
		color: var(--clay);
	}
	.meta {
		font: 500 11px var(--mono);
		letter-spacing: 0.06em;
		text-transform: uppercase;
		opacity: 0.7;
	}
</style>
