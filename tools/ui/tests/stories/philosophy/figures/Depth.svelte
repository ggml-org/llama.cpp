<!--
	A little shape, each with its job: the real surfaces of Llama UI side by side, labelled by what their shape tells
	you. A raised button (elevated-control) to press, a recessed field (the Input's matte-well) to type in, a lifted
	card (card-surface, matte glass) to read as one thing, and a floating menu (matte-float) over the page.
-->
<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
</script>

<div class="depth" inert>
	<figure class="item">
		<div class="stage">
			<Button variant="outline">Download</Button>
		</div>

		<figcaption><b>Raised</b> Something to press</figcaption>
	</figure>

	<figure class="item">
		<div class="stage">
			<Input aria-label="Search" placeholder="Search" />
		</div>

		<figcaption><b>Recessed</b> Somewhere to type</figcaption>
	</figure>

	<figure class="item">
		<div class="stage">
			<div class="card card-surface rounded-card">
				<b>Qwen3.8 27B</b>

				<span>Q4_K_M · 16.8 GB</span>
			</div>
		</div>

		<figcaption><b>Lifted</b> One thing to read</figcaption>
	</figure>

	<figure class="item">
		<div class="stage">
			<ul class="menu matte-float rounded-panel">
				<li>Rename</li>

				<li>Duplicate</li>

				<li>Delete</li>
			</ul>
		</div>

		<figcaption><b>Floating</b> Above the page, for now</figcaption>
	</figure>
</div>

<style>
	.depth {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 8px;
	}

	.item {
		display: grid;
		grid-template-rows: 1fr auto;
		gap: 12px;
		margin: 0;
	}

	.stage {
		display: grid;
		place-items: center;
		min-height: 150px;
		padding: 16px;
		border-radius: 12px;
		background: var(--surface-page);
	}

	.card {
		display: grid;
		gap: 2px;
		padding: 12px 14px;
		font-size: 14px;
	}

	.card span {
		color: var(--figure-muted);
		font-size: 12px;
	}

	.menu {
		display: grid;
		min-width: 9rem;
		margin: 0;
		padding: 4px;
		border-width: 1px;
		list-style: none;
		font-size: 13px;
	}

	.menu li {
		padding: 6px 8px;
		border-radius: 6px;
	}

	.menu li:first-child {
		background: color-mix(in oklab, var(--fg-default) 6%, transparent);
	}

	figcaption {
		color: var(--figure-muted);
		font-size: 13px;
		line-height: 1.4;
	}

	figcaption b {
		display: block;
		color: var(--fg-default);
		font-weight: 600;
	}

	@media (width < 720px) {
		.depth {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}
</style>
