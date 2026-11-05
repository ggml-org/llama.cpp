<!--
	Proposal: the lockup's icon-to-wordmark proportion, in even steps from today's lockup (ggml-org/llama.brand,
	build_lockups.py: the icon 1.91× the cap height, 0.81 cap heights before the first letter) to the icon as a glyph of
	Llama Sans (U+E000: exactly the cap height, then its sidebearing and a word space, 0.47 cap heights). The wordmark
	keeps its size, set live in Llama Sans Bold at its Display size; the icon stays centred on the capitals, which at the
	glyph's end puts it on the baseline, as the font draws it.
-->
<script lang="ts">
	import { PATHS, SIZE } from '../../../brand/logo-showcase/lib/logo';

	const CAP = 710; // Llama Sans's cap height, in font units (the wordmark is set at 1000)
	const FROM = { gap: 221.5 / 272.6, icon: 520 / 272.6 }; // today's lockup
	const TO = { gap: (49 + 213 + 69) / CAP, icon: 1 }; // the glyph: its right sidebearing, a space, the L's left
	const STEPS = 6;
	// one scale for every step (px per font unit), so the wordmark is the same size throughout and only the icon and
	// the gap change
	const SCALE = 0.06;

	// the steps kept from the six even ones between them, and the glyph itself at the end
	const KEPT = [1, 3, 4, 5];

	const steps = Array.from({ length: STEPS }, (_, i) => {
		const t = i / (STEPS - 1);

		return {
			gap: FROM.gap + (TO.gap - FROM.gap) * t,
			icon: FROM.icon + (TO.icon - FROM.icon) * t,
			label: i === STEPS - 1 ? 'The glyph' : `Step ${i}`
		};
	}).filter((_, i) => KEPT.includes(i));

	const marks = [
		{ first: 59, name: 'llama.cpp', split: 5 },
		{ first: 69, name: 'Llama', split: 5 }
	];

	const tones = [
		{ field: '#fafafa', ink: '#111111', label: 'light' },
		{ field: '#111111', ink: '#fafafa', label: 'dark' }
	];

	// the wordmark sits on a baseline at 0; the icon's box, centred on the capitals (their middle at -CAP/2)
	function layout(step: { gap: number; icon: number }, first: number) {
		const side = step.icon * CAP;
		const top = -CAP / 2 - side / 2;

		return {
			iconTransform: `translate(0 ${top}) scale(${side / SIZE})`,
			// the first letter's ink starts after the gap; text is placed by its origin, before the sidebearing
			textX: side + step.gap * CAP - first,
			top
		};
	}

	// each wordmark's width, measured once it's laid out, for the viewBox
	let widths = $state<Record<string, number>>({});

	function measure(node: SVGTextElement, key: string) {
		const update = () => (widths[key] = node.getBBox().width + node.getBBox().x);

		document.fonts.ready.then(update);

		return { update };
	}
</script>

<div class="mx-auto grid w-[min(72rem,calc(100%-2rem))] gap-10 py-10">
	<header class="grid gap-2">
		<h1 class="text-3xl font-bold tracking-tight">Lockup proposal</h1>

		<p class="max-w-[64ch] text-sm text-muted-foreground">
			Steps 1, 3 and 4 of {STEPS} even steps between today’s lockup (the icon 1.91× the cap height, a 0.81
			cap gap) and the icon as a glyph of Llama Sans (1× the cap height, a 0.47 gap), and the glyph itself.
			The wordmark is the same size in all of them, and the icon is centred on the capitals.
		</p>
	</header>

	{#each steps as step, i (i)}
		<section class="grid gap-3">
			<div class="flex flex-wrap items-baseline gap-x-3 gap-y-1">
				<h2 class="text-lg font-semibold">{step.label}</h2>

				<p class="font-mono text-xs text-muted-foreground">
					icon {step.icon.toFixed(2)}× cap · gap {step.gap.toFixed(2)} cap
				</p>
			</div>

			<!-- one box per lockup: llama.cpp and Llama, each on light and on dark -->
			<div class="grid gap-3 md:grid-cols-2">
				{#each marks as mark (mark.name)}
					{#each tones as tone (tone.label)}
						{@const l = layout(step, mark.first)}
						{@const key = `${i}-${tone.label}-${mark.name}`}
						{@const top = Math.min(l.top, -CAP - 60)}
						{@const bottom = Math.max(l.top + step.icon * CAP, 230)}
						<figure class="m-0 grid gap-2">
							<div
								style:background={tone.field}
								class="grid min-h-40 place-items-center rounded-card border border-border/60 px-[6%] py-10"
							>
								<svg
									style:width="{((widths[key] ?? 4000) + 20) * SCALE}px"
									class="block h-auto max-w-full"
									viewBox="0 {top} {(widths[key] ?? 4000) + 20} {bottom - top}"
								>
									<g fill={tone.ink} transform={l.iconTransform}>
										{#each PATHS as d (d)}<path {d} />{/each}
									</g>

									<text
										use:measure={key}
										fill={tone.ink}
										font-family="'Llama Sans'"
										font-size="1000"
										font-weight="700"
										x={l.textX}
										y="0"
										>{mark.name.slice(0, mark.split)}<tspan fill="var(--art-cinder)"
											>{mark.name.slice(mark.split)}</tspan
										></text
									>
								</svg>
							</div>

							<figcaption class="text-xs text-muted-foreground">
								{mark.name}, on {tone.label}
							</figcaption>
						</figure>
					{/each}
				{/each}
			</div>
		</section>
	{/each}
</div>
