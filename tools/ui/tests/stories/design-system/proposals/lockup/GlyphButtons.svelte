<!--
	Proposal: the icon in buttons at four sizes, from the font's own glyph (U+E000, 1× the cap height) to emoji-like
	ones that fill most of the line, so the right weight for an icon in a control can be judged. The bigger ones are
	drawn, centred on the button's line box, which Llama Sans centres on the cap height.
-->
<script lang="ts">
	import { PATHS, SIZE } from '../../../brand/logo-showcase/lib/logo';
	import { Button } from '$lib/components/ui/button';

	const CAP = 0.71; // the cap height, in em

	const sizes = [
		{ label: 'The glyph', note: '1× the cap height, typed', ratio: 1, typed: true },
		{ label: 'Step 4', note: '1.18× the cap height', ratio: 1.18 },
		{ label: 'Emoji-like', note: '1.4× the cap height, about 1 em', ratio: 1.4 },
		{ label: 'Emoji, large', note: '1.6× the cap height, as a colour emoji sits', ratio: 1.6 }
	];

	const buttons = [
		{ label: 'Open in Llama', size: 'sm', variant: 'default' },
		{ label: 'Open in Llama', size: 'default', variant: 'default' },
		{ label: 'Open in Llama', size: 'lg', variant: 'default' },
		{ label: 'llama.app', size: 'default', variant: 'outline' },
		{ label: 'Run with llama.cpp', size: 'default', variant: 'secondary' }
	] as const;
</script>

<div class="mx-auto grid w-[min(72rem,calc(100%-2rem))] gap-10 py-10">
	<header class="grid gap-2">
		<h1 class="text-3xl font-bold tracking-tight">Glyph in buttons</h1>

		<p class="max-w-[64ch] text-sm text-muted-foreground">
			The icon in the design system’s buttons at four sizes: the font’s own glyph, step 4’s
			proportion, and two emoji-like ones. Each is centred on the line, like the label’s capitals.
		</p>
	</header>

	{#each sizes as s (s.label)}
		<section class="grid gap-3">
			<div class="flex flex-wrap items-baseline gap-x-3 gap-y-1">
				<h2 class="text-lg font-semibold">{s.label}</h2>

				<p class="font-mono text-xs text-muted-foreground">{s.note}</p>
			</div>

			<div class="flex flex-wrap items-center gap-3 rounded-card border border-border/60 p-6">
				{#each buttons as b, i (i)}
					<Button size={b.size} variant={b.variant}>
						{#if s.typed}
							<span></span>
						{:else}
							<svg
								style:height="{s.ratio * CAP}em"
								style:width="{s.ratio * CAP}em"
								aria-hidden="true"
								class="shrink-0"
								viewBox="0 0 {SIZE} {SIZE}"
							>
								<g fill="currentColor"
									>{#each PATHS as d (d)}<path {d} />{/each}</g
								>
							</svg>
						{/if}

						{b.label}
					</Button>
				{/each}
			</div>
		</section>
	{/each}
</div>
