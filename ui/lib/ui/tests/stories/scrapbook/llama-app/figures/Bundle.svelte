<!--
	"Nothing to bundle": three apps without Llama, each shipping its own engine and its own copy of the model, and the
	same three with Llama, wired down to one engine and one model under it. Each side's kicker carries its disk use.
	With `arch`, the apps are cinder, the engine and model grey and Llama's slab its sky, with Architecture's key.
-->
<script lang="ts">
	import { Logo } from '$lib/components/app/misc';

	let { arch = false }: { arch?: boolean } = $props();

	const apps = ['Chat app', 'Coding agent', 'Your app'];
</script>

{#snippet layers()}
	<span class="layer"><span>Engine</span></span>
	<span class="layer"><span>Model</span><code>19 GB</code></span>
{/snippet}

<div class={['lf', { arch }]}>
	<div class="sides">
		<figure class="panel">
			<figcaption class="kicker"><b>Without Llama</b><span>57 GB on disk</span></figcaption>

			<div class="cols">
				{#each apps as a (a)}
					<div class="col">
						<span class="part app name">{a}</span>
						<span class="wire"></span>
						<span class="part be stack">{@render layers()}</span>
					</div>
				{/each}
			</div>
		</figure>

		<figure class="panel">
			<figcaption class="kicker"><b>With Llama</b><span>19 GB on disk</span></figcaption>

			<div class="cols">
				{#each apps as a (a)}
					<div class="col">
						<span class="part app name">{a}</span>
						<span class="wire"></span>
					</div>
				{/each}
			</div>

			<div class="part lib slab">
				<span class="who"><Logo style="--size: 12px" />Llama</span>
				<span class="stack">{@render layers()}</span>
			</div>
		</figure>
	</div>

	{#if arch}
		<p class="key">
			<span><i style:border-color="var(--app-line)" style:background="var(--app-bg)"></i>Apps</span>
			<span><i style:border-color="var(--lib-line)" style:background="var(--lib-bg)"></i>Llama</span>
			<span><i style:border-color="var(--be-line)" style:background="var(--be-bg)"></i>Engine and model</span>
		</p>
	{/if}
</div>

<style>
	.lf {
		display: grid;
		gap: 12px;
	}
	.sides {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(19rem, 1fr));
		gap: 12px;
	}
	figure {
		align-content: start;
		margin: 0;
	}
	figcaption {
		justify-content: space-between;
	}
	.cols {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 8px;
	}
	.col {
		display: grid;
		justify-items: stretch;
	}
	.lf .name {
		justify-content: center;
		padding: 8px 4px;
		font-size: 12px;
		text-align: center;
	}
	.lf.arch .name {
		color: var(--app);
	}
	.col .wire {
		height: 14px;
	}
	.lf .stack {
		display: grid;
		gap: 0;
		padding: 0 10px;
		font-size: 12px;
	}
	.layer {
		display: flex;
		flex-wrap: wrap;
		gap: 2px 8px;
		justify-content: space-between;
		padding-block: 9px;
	}
	.layer + .layer {
		border-top: 1px solid var(--line-default);
	}
	.lf.arch .layer {
		color: var(--be);
	}
	.layer code {
		font-size: 11px;
	}
	/* one engine and one model, under all three: Llama's slab */
	.lf .slab {
		display: flex;
		gap: 0;
		align-items: stretch;
		margin-top: -16px;
		padding: 0;
	}
	.who {
		display: flex;
		gap: 6px;
		align-items: center;
		margin-block: 10px;
		padding: 0 14px;
		border-right: 1px solid var(--lib-line);
		color: var(--lib);
		font-size: 12px;
		font-weight: 600;
	}
	.lf .slab .stack {
		flex: 1;
	}
	.lf.arch .slab .layer + .layer {
		border-color: var(--lib-line);
	}
</style>
