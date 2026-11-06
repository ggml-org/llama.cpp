<!--
	How it shows: the same ideas in the product, the design and communications, each as a real piece. A model row as
	the app shows it, the neutral greys with one cinder mark, and llama.app's link preview.
-->
<script lang="ts">
	import LinkPreview from '../../brand/communications/posts/LinkPreview.svelte';
	import { linkPreview } from '../../brand/communications/posts/post';
	import { Badge } from '$lib/components/ui/badge';

	const greys = [
		'--neutral-50',
		'--neutral-150',
		'--neutral-300',
		'--neutral-500',
		'--neutral-700',
		'--neutral-900'
	];
</script>

<div class="shows">
	<section class="card card-surface rounded-card">
		<span class="label">In the product</span>

		<div class="model rounded-panel bg-surface-lifted">
			<p class="name">Your model <Badge variant="outline">Q4_K_M</Badge></p>

			<p class="where"><i class="live"></i>On this computer · <code>localhost:8080</code></p>
		</div>

		<p class="note">Which model, on which server, always in view</p>
	</section>

	<section class="card card-surface rounded-card">
		<span class="label">In the design</span>

		<div class="greys">
			{#each greys as g (g)}<i style:background="var({g})"></i>{/each}<i class="cinder"></i>
		</div>

		<p class="note">Neutral greys, and one accent</p>
	</section>

	<section class="card card-surface rounded-card">
		<span class="label">In communications</span>

		<LinkPreview copy={linkPreview} kind="og" tone="hero" />

		<p class="note">The real app, in a calm voice</p>
	</section>
</div>

<style>
	.shows {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr));
		gap: 8px;
		align-items: start;
	}

	.card {
		display: grid;
		gap: 12px;
		padding: 16px;
	}

	.label {
		color: var(--figure-muted);
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	p {
		margin: 0;
	}

	.model {
		display: grid;
		gap: 4px;
		padding: 10px 12px;
	}

	.name {
		display: flex;
		gap: 8px;
		align-items: center;
		font-weight: 600;
	}

	.where,
	.note {
		color: var(--figure-muted);
		font-size: 13px;
	}

	.where {
		display: flex;
		flex-wrap: wrap;
		gap: 2px 6px;
		align-items: center;
	}

	code {
		font: 500 12px var(--mono);
	}

	.live {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: var(--art-green);
	}

	.greys {
		display: grid;
		grid-template-columns: repeat(7, 1fr);
		height: 56px;
		overflow: hidden;
		border-radius: 10px;
		box-shadow: inset 0 0 0 1px var(--line-default);
	}

	.cinder {
		background: var(--art-cinder);
	}
</style>
