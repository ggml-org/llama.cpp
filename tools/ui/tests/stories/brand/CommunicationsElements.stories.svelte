<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import AppPanel from './communications/posts/AppPanel.svelte';
	import BranchMotif from './communications/posts/BranchMotif.svelte';
	import Command from './communications/posts/Command.svelte';
	import Curve from './communications/posts/Curve.svelte';
	import DotGrid from './communications/posts/DotGrid.svelte';
	import Figure from './communications/posts/Figure.svelte';
	import type { Ground } from './communications/posts/post';
	import PostChip from './communications/posts/PostChip.svelte';
	import PostFooter from './communications/posts/PostFooter.svelte';
	import PostFrame from './communications/posts/PostFrame.svelte';
	import PostHeader from './communications/posts/PostHeader.svelte';
	import QuantBlocks from './communications/posts/QuantBlocks.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Communications' elements: the smallest parts of a piece, each in units, on a tile of the grounds. Lockups and the
	// header, chips and labels, figures, commands, motifs and the app itself
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Communications/Elements'
	});

	const tile = { height: 720, name: 'Tile', width: 1080 };
	const tones: Ground[] = ['light', 'dark', 'day'];
</script>

<Story asChild name="Header and footer">
	<StateGallery minItemWidth="20rem">
		{#each ['light', 'dark'] as const as tone (tone)}
			<StateGalleryItem
				label="Header and footer, {tone}"
				note="The top and bottom rows of every piece"
			>
				<PostFrame format={tile} label="Header and footer" {tone}>
					<div class="stack">
						<PostHeader brand="Llama" url="llama.app" />

						<PostHeader brand="llama.cpp" url="github.com/ggml-org/llama.cpp" />

						<PostHeader brand="icon" kicker="From the llama.cpp team" url="llama.app/models" />

						<PostHeader brand="Llama" pill="New models available today" />
					</div>

					<PostFooter left="By the llama.cpp team and Hugging Face" right="Free · Open source" />
				</PostFrame>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>

<Story asChild name="Chips and labels">
	<StateGallery minItemWidth="20rem">
		{#each tones as tone (tone)}
			<StateGalleryItem
				label="Chips and labels, {tone}"
				note="Outline for use cases, solid for a tag, cinder once"
			>
				<PostFrame format={tile} label="Chips" {tone}>
					<div class="stack">
						<p class="t-label">Fresh drops</p>

						<div class="row">
							<PostChip>route a ticket</PostChip><PostChip>moderate content</PostChip>
						</div>

						<div class="row">
							<PostChip variant="solid">Every modality</PostChip><PostChip variant="accent"
								>Index leader</PostChip
							>
						</div>
					</div>
				</PostFrame>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>

<Story asChild name="Figures and commands">
	<StateGallery minItemWidth="20rem">
		{#each ['light', 'dark'] as const as tone (tone)}
			<StateGalleryItem
				label="Figures and commands, {tone}"
				note="Real numbers in tabular figures; the prompt in cinder"
			>
				<PostFrame format={tile} label="Figures" {tone}>
					<div class="row figures">
						<Figure accent label="tokens per second" size={3.6} value="142" />

						<Figure label="GB at Q4_K_M" size={3.6} value="6.6" />
					</div>

					<Command command="llama-server -hf ggml-org/gemma-4-E4B-it-GGUF" />
				</PostFrame>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>

<Story asChild name="Motifs">
	<StateGallery minItemWidth="16rem">
		{#each ['light', 'dark'] as const as tone (tone)}
			<StateGalleryItem label="Token tree, {tone}" note="Decisions, sampling, probabilities">
				<PostFrame format={tile} label="Token tree" {tone}>
					<div class="motif fill"><BranchMotif /></div>
				</PostFrame>
			</StateGalleryItem>

			<StateGalleryItem label="Curve, {tone}" note="Speed, benchmarks, anything over time">
				<PostFrame format={tile} label="Curve" {tone}>
					<div class="motif fill"><Curve /></div>
				</PostFrame>
			</StateGalleryItem>

			<StateGalleryItem label="Quantised blocks, {tone}" note="Quants, sizes, fitting in memory">
				<PostFrame format={tile} label="Quantised blocks" {tone}>
					<div class="blocks"><QuantBlocks columns={12} rows={6} /></div>
				</PostFrame>
			</StateGalleryItem>

			<StateGalleryItem
				label="Dot grid, {tone}"
				note="The unit made visible, behind a word or figure"
			>
				<PostFrame format={tile} label="Dot grid" {tone}>
					<DotGrid at={[9, 5]} />
				</PostFrame>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>

<Story asChild name="The app">
	<StateGallery minItemWidth="20rem">
		{#each ['day', 'night'] as const as tone (tone)}
			<StateGalleryItem
				label="Menu bar and panel, {tone}"
				note="The Llama app as it is, at its own size"
			>
				<PostFrame format={{ height: 720, name: 'Tile', width: 720 }} label="The app" {tone}>
					<div class="app"><AppPanel /></div>
				</PostFrame>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>

<style>
	.stack {
		display: grid;
		gap: calc(var(--u) * 1.2);
		margin-bottom: auto;
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		gap: calc(var(--u) * 0.3);
	}
	.figures {
		gap: calc(var(--u) * 3);
		margin-bottom: auto;
	}
	.motif {
		flex: 1;
	}
	.blocks {
		margin-block: auto;
		padding-inline: calc(var(--u) * 2);
	}
	.app {
		display: grid;
		flex: 1;
		place-items: center;
	}
</style>
