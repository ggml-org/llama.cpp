<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import {
		type FormatKey,
		FORMATS,
		grid,
		type Ground,
		GROUNDS,
		type PostFormat
	} from './communications/posts/post';
	import PostChip from './communications/posts/PostChip.svelte';
	import PostFrame from './communications/posts/PostFrame.svelte';
	import PostHeader from './communications/posts/PostHeader.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Communications' foundations: the constraints every piece is built on. The formats and their grid (one unit, a
	// 24th of the short side), the grounds a piece is set on, and the type scale in units
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Communications/Foundations'
	});

	const formats = Object.keys(FORMATS) as FormatKey[];
	const grounds: Ground[] = ['light', 'dark', 'sand', 'hero', 'navy'];
	const tile = { height: 720, name: 'Tile', width: 1080 };
	const scale = [
		['t-display', 'Display', '3.2 u', 'Your AI.'],
		['t-headline', 'Headline', '2 u', 'On your computer.'],
		[
			't-lede',
			'Lede',
			'0.72 u',
			'Run open models on your own hardware, from a laptop to a server.'
		],
		['t-body', 'Body', '0.55 u', 'Cards, notes and footers, in Llama Sans.'],
		['t-label', 'Label', '0.48 u Mono', 'Fresh drops'],
		['t-mono', 'Mono', '0.5 u Mono', 'ggml-org/gemma-4-E4B-it-GGUF · 4.59 GB']
	];
</script>

<Story asChild name="Formats and grid">
	<StateGallery minItemWidth="18rem">
		{#each formats as key (key)}
			{@const f = FORMATS[key] as PostFormat}
			{@const g = grid(f)}
			<StateGalleryItem label={f.name} note="{f.width} × {f.height} · {f.use}">
				<PostFrame format={f} label={f.name} overlay tone="light">
					<div class="spec">
						<p class="t-headline">{f.width} × {f.height}</p>

						<p class="t-mono">
							u = {g.unit.toFixed(1)} px · margin 2 u · {g.columns} columns · gutter ½ u · {g.shape}
						</p>

						{#if f.safe}<p class="t-mono">
								safe area: top {f.safe.top}, bottom {f.safe.bottom}
							</p>{/if}
					</div>
				</PostFrame>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>

<Story asChild name="Grounds">
	<StateGallery minItemWidth="18rem">
		{#each grounds as ground (ground)}
			<StateGalleryItem label={GROUNDS[ground].name} note={GROUNDS[ground].note}>
				<PostFrame format={tile} label={GROUNDS[ground].name} tone={ground}>
					<PostHeader brand="Llama" size={1.2} url="llama.app" />

					<p class="t-headline grounded">Your AI.<br />On your computer.</p>

					<div class="chips">
						<PostChip>outline</PostChip><PostChip variant="solid">solid</PostChip><PostChip
							variant="accent">accent</PostChip
						>
					</div>

					<p class="t-label">Lockups: {GROUNDS[ground].lockups}</p>
				</PostFrame>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>

<Story asChild name="Type">
	<StateGallery columns={1}>
		{#each ['light', 'dark'] as const as tone (tone)}
			<StateGalleryItem
				label="Type scale, {tone}"
				note="In units of the format: the landscape's is 37.5 px"
			>
				<PostFrame format={FORMATS.landscape} label="Type scale" {tone}>
					<div class="scale">
						{#each scale as [cls, name, size, sample] (cls)}
							<p class="t-label">{name}<br />{size}</p>

							<p class={cls}>{sample}</p>
						{/each}
					</div>
				</PostFrame>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>

<style>
	.spec {
		display: grid;
		gap: calc(var(--u) * 0.6);
		margin-top: auto;
	}
	.grounded {
		margin-top: auto;
	}
	.chips {
		display: flex;
		gap: calc(var(--u) * 0.3);
		margin-block: calc(var(--u) * 1) calc(var(--u) * 1.2);
	}
	.scale {
		display: grid;
		grid-template-columns: calc(var(--u) * 4) 1fr;
		align-items: baseline;
		gap: calc(var(--u) * 0.6) var(--u);
		margin-block: auto;
	}
</style>
