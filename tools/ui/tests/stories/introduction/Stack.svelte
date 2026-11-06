<!--
	What this Storybook holds, as building blocks: the Llama Brand at the base, Llama UI built on it, Llama Chat built
	from Llama UI, and the communications made from the brand and the app beside them. Llama Chat, the one people use,
	carries the cinder mark; the brand carries the icon.
-->
<script lang="ts">
	import { PATHS, SIZE } from '../brand/logo-showcase/lib/logo';
</script>

<div class="stack">
	<div class="block chat card-surface rounded-card">
		<span class="n">03</span>

		<h3>Llama Chat</h3>

		<p>The app, built from Llama UI</p>
	</div>

	<div class="block comms card-surface rounded-card">
		<h3>Communications</h3>

		<p>Posts, store screenshots, link previews and swag, made from the brand and the app</p>
	</div>

	<div class="block ui card-surface rounded-card">
		<span class="n">02</span>

		<h3>Llama UI</h3>

		<p>The components and their tokens</p>
	</div>

	<div class="block brand card-surface rounded-card">
		<span class="n">01</span>

		<h3>
			<svg aria-hidden="true" class="icon" viewBox="0 0 {SIZE} {SIZE}">
				{#each PATHS as d (d)}<path {d} />{/each}
			</svg>

			Llama Brand
		</h3>

		<p>The icon, Llama Sans and Llama Mono, the colours</p>
	</div>
</div>

<style>
	.stack {
		display: grid;
		grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
		grid-template-areas:
			'chat comms'
			'ui comms'
			'brand brand';
		gap: 8px;
	}

	.block {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr);
		gap: 4px 12px;
		align-content: center;
		align-items: baseline;
		padding: 16px 20px;
	}

	.chat {
		grid-area: chat;
	}

	.ui {
		grid-area: ui;
	}

	.brand {
		grid-area: brand;
	}

	.brand h3 {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.comms {
		grid-area: comms;
		grid-template-columns: minmax(0, 1fr);
	}

	.n {
		color: var(--figure-muted);
		font: 500 12px var(--mono);
	}

	/* the one accent: the app people use */
	.chat .n {
		color: var(--art-cinder);
	}

	.icon {
		width: 18px;
		height: 18px;
		fill: currentColor;
	}

	h3 {
		margin: 0;
		font-size: 18px;
		font-weight: 600;
	}

	p {
		grid-column: 2;
		margin: 0;
		color: var(--figure-muted);
		font-size: 15px;
		line-height: 1.45;
	}

	.comms p {
		grid-column: 1;
	}

	@media (max-width: 640px) {
		.stack {
			grid-template-columns: minmax(0, 1fr);
			grid-template-areas: 'chat' 'ui' 'brand' 'comms';
		}
	}
</style>
