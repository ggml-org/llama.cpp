<!--
	"Nothing to set up": the settings Llama picks for a model, each with how it picks it. The model heads the figure, the
	kicker says who chose (with cinder's dot, or, with `arch`, Llama's sky), and every row ends in an Auto badge.
-->
<script lang="ts">
	import { Gauge, Image, Layers, ScrollText, Zap } from '@lucide/svelte';
	import qwen from '$lib/assets/model-logos/qwen.svg?raw';
	import { Badge } from '$lib/components/ui/badge';

	let { arch = false }: { arch?: boolean } = $props();

	const tuned = [
		{ how: 'Highest precision that fits in memory', icon: Gauge, setting: 'Quantization' },
		{ how: 'Only sizes that fit in memory', icon: ScrollText, setting: 'Context size' },
		{ how: 'On for models that understand images', icon: Image, setting: 'Image input' },
		{ how: 'On for models that support it', icon: Zap, setting: 'Speculative decoding' },
		{ how: 'Larger on Macs with 32 GB or more', icon: Layers, setting: 'Batch size' }
	];
</script>

<div class={['lf', { arch }]}>
	<div class="panel">
		<header>
			<span class="model"><span class="mark">{@html qwen}</span>Qwen3.8<Badge variant="outline">27B</Badge></span>
			<p class="kicker">
				{#if arch}<i class="sky"></i>{:else}<i class="dot"></i>{/if}Chosen by Llama, for your Mac
			</p>
		</header>

		<ul class="part">
			{#each tuned as t (t.setting)}
				<li>
					<span class="icon"><t.icon size={16} /></span>
					<span class="text"><b>{t.setting}</b><span class="muted">{t.how}</span></span>
					<Badge class={arch ? 'auto' : ''} variant="outline">Auto</Badge>
				</li>
			{/each}
		</ul>
	</div>
</div>

<style>
	header {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 16px;
		align-items: center;
		justify-content: space-between;
	}
	.model {
		display: inline-flex;
		gap: 8px;
		align-items: center;
		font-weight: 500;
	}
	.sky {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: var(--lib);
	}
	ul {
		gap: 0;
		margin: 0;
		padding: 0 14px;
		list-style: none;
	}
	li {
		display: flex;
		gap: 12px;
		align-items: center;
		padding-block: 11px;
	}
	li + li {
		border-top: 1px solid var(--line-default);
	}
	.icon {
		display: grid;
		flex: none;
		place-items: center;
		width: 32px;
		height: 32px;
		border-radius: var(--radius-control);
		background: var(--surface-muted);
		color: var(--figure-muted);
	}
	.text {
		display: grid;
		flex: 1;
		min-width: 0;
	}
	.text b {
		font-weight: 500;
	}
	/* Llama's choice, in its sky */
	li :global(.auto) {
		border-color: var(--lib-line);
		background: var(--lib-bg);
		color: var(--lib);
	}
</style>
