<!--
	Simple by default, powerful on request, as two columns built from real Llama UI pieces. Starting out: the
	recommended model with one Download button, and a model found by name in the global search next to conversations
	and projects. Power user: the model manager and Discover as switches, off by default, and what opens up once they
	are on. The recommended model is not named: its row carries the Llama icon.
-->
<script lang="ts">
	import { PATHS, SIZE } from '../../brand/logo-showcase/lib/logo';
	import { Box, Folder, MessageSquare, Search } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { Switch } from '$lib/components/ui/switch';

	const results = [
		{ icon: Box, kind: 'Model', name: 'Qwen3.8 27B' },
		{ icon: MessageSquare, kind: 'Conversation', name: 'Qwen or gpt-oss for code?' },
		{ icon: Folder, kind: 'Project', name: 'Qwen experiments' }
	];

	const options = [
		'Any GGUF',
		'Q8_0 … Q2_K',
		'Sampling',
		'Penalties',
		'System prompts',
		'Recipes',
		'Server flags'
	];
</script>

<div class="paths">
	<section class="path card-surface rounded-card">
		<p class="who first">Starting out</p>

		<div class="pick rounded-panel bg-surface-lifted elevated-control">
			<svg aria-hidden="true" class="icon" viewBox="0 0 {SIZE} {SIZE}">
				{#each PATHS as d (d)}<path {d} />{/each}
			</svg>

			<div class="pick-text">
				<b>Recommended</b>

				<span>Reads, sees and hears ∣ 1.8 GB</span>
			</div>

			<Button size="sm" tabindex={-1}>Download</Button>
		</div>

		<div class="search rounded-panel bg-surface-lifted elevated-control" inert>
			<p class="query"><Search aria-hidden="true" size={14} /> qwen</p>

			<ul>
				{#each results as r (r.name)}
					<li>
						<r.icon aria-hidden="true" size={14} />

						<span class="name">{r.name}</span>

						<span class="kind">{r.kind}</span>
					</li>
				{/each}
			</ul>
		</div>

		<p class="note">One model picked for the device, one click. Others found by name.</p>
	</section>

	<section class="path card-surface rounded-card">
		<p class="who">Power user</p>

		<div class="switches rounded-panel bg-surface-lifted elevated-control" inert>
			<p class="row">
				<span>Model manager</span>

				<Switch checked />
			</p>

			<p class="row">
				<span>Discover</span>

				<Switch checked />
			</p>

			<p class="off">Both off by default</p>
		</div>

		<div class="options">
			{#each options as option (option)}
				<Badge variant="outline">{option}</Badge>
			{/each}
		</div>

		<p class="note">Turned on in Settings. Everything else is one step away.</p>
	</section>
</div>

<style>
	.paths {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 8px;
	}

	.path {
		display: grid;
		grid-template-rows: auto auto 1fr auto;
		gap: 10px;
		padding: 16px;
	}

	p {
		margin: 0;
	}

	.who {
		color: var(--figure-muted);
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.who.first {
		color: var(--art-cinder);
	}

	.pick {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr) auto;
		align-items: center;
		gap: 10px;
		padding: 10px 12px;
	}

	.icon {
		width: 22px;
		height: 22px;
		fill: currentColor;
	}

	.pick-text {
		display: grid;
	}

	b {
		font-size: 14px;
		font-weight: 600;
	}

	.pick-text span,
	.kind,
	.off {
		color: var(--figure-muted);
		font-size: 12px;
	}

	.search {
		display: grid;
		gap: 6px;
		padding: 10px 12px;
	}

	.query {
		display: flex;
		align-items: center;
		gap: 6px;
		padding-bottom: 6px;
		border-bottom: 1px solid var(--line-default);
		font-size: 13px;
	}

	.search ul {
		display: grid;
		gap: 6px;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.search li {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr) auto;
		align-items: center;
		gap: 8px;
		margin: 0;
		font-size: 13px;
	}

	.name {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.switches {
		display: grid;
		gap: 8px;
		padding: 10px 12px;
	}

	.row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		font-size: 13px;
	}

	.options {
		display: flex;
		flex-wrap: wrap;
		align-content: start;
		gap: 6px;
	}

	.note {
		color: var(--figure-muted);
		font-size: 13px;
		line-height: 1.45;
	}

	@media (width < 640px) {
		.paths {
			grid-template-columns: minmax(0, 1fr);
		}
	}
</style>
