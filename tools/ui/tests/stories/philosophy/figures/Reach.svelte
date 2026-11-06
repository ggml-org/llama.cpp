<!--
	Ordinary hardware: the machines people already have, from a laptop to a server, each with its memory as a bar and
	the share a model takes at a 4-bit quant. Sizes are rough, to show the range, not a promise. The laptop, where most
	people are, is the cinder one.
-->
<script lang="ts">
	const machines = [
		{ fits: 'Small models, up to about 8B', memory: 8, model: 5.5, name: 'Laptop' },
		{ fits: 'Up to about 20B', memory: 16, model: 12, name: 'Desktop' },
		{ fits: 'Up to about 70B', memory: 64, model: 42, name: 'Workstation' },
		{ fits: 'The largest open models', memory: 256, model: 200, name: 'Server' }
	];
</script>

<div class="reach">
	{#each machines as m, i (m.name)}
		<section class:first={i === 0} class="machine card-surface rounded-card">
			<h3>{m.name}</h3>

			<p class="memory"><code>{m.memory} GB</code> of memory</p>

			<div aria-hidden="true" class="bar">
				<i style:width="{(m.model / m.memory) * 100}%"></i>
			</div>

			<p class="fits">{m.fits}</p>
		</section>
	{/each}
</div>

<style>
	.reach {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr));
		gap: 8px;
	}

	.machine {
		display: grid;
		gap: 8px;
		align-content: start;
		padding: 16px;
	}

	h3 {
		margin: 0;
		font-size: 17px;
		font-weight: 600;
	}

	p {
		margin: 0;
		font-size: 13px;
		line-height: 1.4;
	}

	.memory,
	.fits {
		color: var(--figure-muted);
	}

	code {
		color: var(--fg-default);
		font: 600 13px var(--mono);
	}

	.bar {
		height: 10px;
		overflow: hidden;
		border-radius: 999px;
		background: color-mix(in oklab, var(--fg-default) 8%, transparent);
	}

	.bar i {
		display: block;
		height: 100%;
		border-radius: inherit;
		background: var(--fg-default);
	}

	/* the one accent: the machine most people have */
	.first .bar i {
		background: var(--art-cinder);
	}
</style>
