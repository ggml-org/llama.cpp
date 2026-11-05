<!--
	The colours and gradients the brand's designs use (the logo, the specimen boards, the posters, the social assets),
	as swatches: the grounds and inks, the gradients, and the colours for shapes and accents, with cinder as the one
	accent orange. Flat colours read the design system's primitives (src/lib/styles/tokens/primitives.css); the
	gradients are the ones the brand pages draw, from llama.app's hero.
-->
<script lang="ts">
	import { primitiveGroups } from '../../design-system/foundations/tokens';

	interface Swatch {
		/** light text on it */
		dark?: boolean;
		name: string;
		note?: string;
		/** a CSS custom property of the design system, or a literal colour or gradient */
		token?: string;
		value?: string;
	}

	// the hero gradient from llama.app and its navy night version (as in the specimen's boards and the social assets)
	const HERO =
		'radial-gradient(ellipse at 15% 10%, #dcecfb 0%, transparent 55%), radial-gradient(ellipse at 90% 95%, #e8f2fc 0%, transparent 50%), linear-gradient(160deg, #c4ddf6 0%, #d0e4f7 55%, #dfecf9 100%)';
	const NAVY =
		'radial-gradient(ellipse at 15% 10%, #22385c 0%, transparent 55%), radial-gradient(ellipse at 90% 95%, #1e3752 0%, transparent 50%), linear-gradient(160deg, #152238 0%, #172a42 55%, #1a3049 100%)';
	const GREYS =
		'linear-gradient(160deg, var(--neutral-400) 0%, var(--neutral-300) 55%, var(--neutral-200) 100%)';
	const DEEP = 'linear-gradient(160deg, var(--neutral-800) 0%, var(--neutral-900) 100%)';

	const groups: { note: string; swatches: Swatch[]; title: string }[] = [
		{
			note: 'Fields stay neutral as a rule: the logo’s ink and light, paper and beige.',
			swatches: [
				{ dark: true, name: 'Ink', note: 'the logo on light, text', value: '#111111' },
				{ name: 'Light', note: 'the logo on dark, its field', value: '#fafafa' },
				{ name: 'Paper', token: 'neutral-50' },
				{ name: 'Beige 50', token: 'beige-50' },
				{ name: 'Beige 100', token: 'beige-100' },
				{ dark: true, name: 'Beige ink', note: 'text on beige', token: 'beige-900' }
			],
			title: 'Grounds and ink'
		},
		{
			note: 'Fields with a purpose: the hero blue and its navy night from llama.app, and the greys.',
			swatches: [
				{ name: 'Hero blue', note: 'llama.app’s hero, ink #0f1f33', value: HERO },
				{ dark: true, name: 'Navy', note: 'the hero at night, white ink', value: NAVY },
				{ name: 'Greys', note: 'a cool title field', value: GREYS },
				{ dark: true, name: 'Deep', note: 'the dark field', value: DEEP }
			],
			title: 'Gradients'
		},
		{
			note: 'For shapes, numerals and accents, never for status and never red. Cinder is the one accent: the .cpp of the logotype and the mark on a detail.',
			swatches: [
				{ dark: true, name: 'Cinder', note: 'the accent · #ee6c1e', token: 'art-cinder' },
				{ name: 'Cinder light', token: 'art-cinder-light' },
				{ dark: true, name: 'Art blue', token: 'art-blue' },
				{ name: 'Art sky', token: 'art-sky' },
				{ dark: true, name: 'Art navy', token: 'art-navy' },
				{ dark: true, name: 'Art green', token: 'art-green' },
				{ name: 'Art amber', token: 'art-amber' }
			],
			title: 'Colours'
		}
	];

	// each primitive's declared value, to print under its swatch
	const declared = new Map(primitiveGroups.flatMap((g) => g.tokens.map((t) => [t.name, t.light])));
	const isGradient = (s: Swatch) => !s.token && s.value?.includes('gradient');
</script>

<div class="mx-auto grid w-[min(72rem,calc(100%-2rem))] gap-12 py-10">
	<header class="grid gap-2">
		<h1 class="text-3xl font-bold tracking-tight">Palette</h1>

		<p class="max-w-[64ch] text-sm text-muted-foreground">
			The colours and gradients the brand’s designs use: the logo, the specimen boards, the posters
			and the social assets. The UI’s own tokens are in UI / Foundations / Colors.
		</p>
	</header>

	{#each groups as group (group.title)}
		<section class="grid gap-4">
			<div class="grid gap-1">
				<h2 class="text-lg font-semibold">{group.title}</h2>

				<p class="max-w-[64ch] text-xs text-muted-foreground">{group.note}</p>
			</div>

			<div class="grid grid-cols-[repeat(auto-fill,minmax(12rem,1fr))] gap-4">
				{#each group.swatches as s (s.name)}
					<figure class="m-0 grid gap-2">
						<div
							style:background={s.token ? `var(--${s.token})` : s.value}
							class={[
								'flex aspect-[4/3] items-end rounded-card border border-border/60 p-3 text-label-sm',
								s.dark ? 'text-white' : 'text-[#111111]'
							]}
						>
							{s.name}
						</div>

						<figcaption class="grid gap-0.5 font-mono text-[11px] leading-4">
							{#if s.token}
								<span class="text-fg-default">--{s.token}</span>

								<span class="truncate text-fg-muted">{declared.get(s.token)}</span>
							{:else}
								<span class="text-fg-default">{isGradient(s) ? 'gradient' : s.value}</span>
							{/if}

							{#if s.note}<span class="text-fg-muted">{s.note}</span>{/if}
						</figcaption>
					</figure>
				{/each}
			</div>
		</section>
	{/each}
</div>
