<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import AppPanel from './communications/posts/AppPanel.svelte';
	import BranchMotif from './communications/posts/BranchMotif.svelte';
	import Command from './communications/posts/Command.svelte';
	import Curve from './communications/posts/Curve.svelte';
	import DotGrid from './communications/posts/DotGrid.svelte';
	import Figure from './communications/posts/Figure.svelte';
	import Gpu from './communications/posts/motifs/Gpu.svelte';
	import GradientDescent from './communications/posts/motifs/GradientDescent.svelte';
	import Laptop from './communications/posts/motifs/Laptop.svelte';
	import NeuralNet from './communications/posts/motifs/NeuralNet.svelte';
	import Server from './communications/posts/motifs/Server.svelte';
	import Tensor from './communications/posts/motifs/Tensor.svelte';
	import VectorSpace from './communications/posts/motifs/VectorSpace.svelte';
	import WorkstationLarge from './communications/posts/motifs/WorkstationLarge.svelte';
	import WorkstationSmall from './communications/posts/motifs/WorkstationSmall.svelte';
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

	// specimen tiles at their real size, on one fixed unit so every part reads the same, sized to what they hold; the
	// galleries fit as many to a row as the width allows
	const U = 28;
	const tile = (width: number, height: number) => ({ height, name: 'Tile', width });
	const rem = (px: number) => `${px / 16}rem`;
	const header = tile(540, 420);
	const chips = tile(352, 340);
	const figures = tile(540, 440);
	const motif = tile(340, 300);
	const app = tile(500, 600);

	// the motifs drawn as SVG, each filling its tile: the ideas, then the machines
	const motifs = [
		{
			component: BranchMotif,
			name: 'Decision tree',
			note: 'Decision models, choices, probabilities'
		},
		{ component: NeuralNet, name: 'Neural network', note: 'Models and what is inside them' },
		{ component: Tensor, name: 'Tensor', note: 'Weights, layers, quantization' },
		{ component: GradientDescent, name: 'Gradient descent', note: 'Training and fine-tuning' },
		{ component: VectorSpace, name: 'Vector space', note: 'Embeddings, search, retrieval' },
		{ component: Curve, name: 'Curve', note: 'Speed, benchmarks, anything over time' },
		{ component: Gpu, name: 'GPU', note: 'Hardware, offload, speed' },
		{ component: Laptop, name: 'Laptop', note: 'The machine people already have' },
		{ component: WorkstationSmall, name: 'Small workstation', note: 'Bigger models on a desk' },
		{
			component: WorkstationLarge,
			name: 'Large workstation',
			note: 'The biggest models on a desk'
		},
		{ component: Server, name: 'Server', note: 'llama-server, sharing, the largest models' }
	];
	const tones: Ground[] = ['light', 'dark', 'day'];
</script>

<Story asChild name="Header and footer">
	<StateGallery minItemWidth={rem(header.width)}>
		{#each ['light', 'dark'] as const as tone (tone)}
			<StateGalleryItem
				label="Header and footer, {tone}"
				note="The top and bottom rows of every piece"
			>
				<PostFrame actual format={header} label="Header and footer" {tone} unit={U}>
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
	<StateGallery minItemWidth={rem(chips.width)}>
		{#each tones as tone (tone)}
			<StateGalleryItem
				label="Chips and labels, {tone}"
				note="Outline for use cases, solid for a tag, cinder once"
			>
				<PostFrame actual format={chips} label="Chips" {tone} unit={U}>
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
	<StateGallery minItemWidth={rem(figures.width)}>
		{#each ['light', 'dark'] as const as tone (tone)}
			<StateGalleryItem
				label="Figures and commands, {tone}"
				note="Real numbers in tabular figures; the prompt in cinder"
			>
				<PostFrame actual format={figures} label="Figures" {tone} unit={U}>
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
	<StateGallery minItemWidth={rem(motif.width)}>
		{#each ['light', 'dark'] as const as tone (tone)}
			{#each motifs as m (m.name)}
				<StateGalleryItem label="{m.name}, {tone}" note={m.note}>
					<PostFrame actual format={motif} label={m.name} {tone} unit={U}>
						<div class="motif fill"><m.component /></div>
					</PostFrame>
				</StateGalleryItem>
			{/each}

			<StateGalleryItem label="Quantised blocks, {tone}" note="Quants, sizes, fitting in memory">
				<PostFrame actual format={motif} label="Quantised blocks" {tone} unit={U}>
					<div class="blocks"><QuantBlocks columns={12} rows={6} /></div>
				</PostFrame>
			</StateGalleryItem>

			<StateGalleryItem
				label="Dot grid, {tone}"
				note="The unit made visible, behind a word or figure"
			>
				<PostFrame actual format={motif} label="Dot grid" {tone} unit={U}>
					<DotGrid at={[6, 4]} />
				</PostFrame>
			</StateGalleryItem>
		{/each}
	</StateGallery>
</Story>

<Story asChild name="The app">
	<StateGallery minItemWidth={rem(app.width)}>
		{#each ['day', 'night'] as const as tone (tone)}
			<StateGalleryItem
				label="Menu bar and panel, {tone}"
				note="The Llama app as it is, at its own size"
			>
				<PostFrame actual format={app} label="The app" {tone} unit={U}>
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
