<!--
	How much of each kind of colour there is: one strip, the greys first and by far, then the status colours, the art
	colours and cinder as slivers, with a legend under it; then the real badges, where the status colours appear and
	nothing else does
-->
<script lang="ts">
	import { Badge } from '$lib/components/ui/badge';

	const greys = [25, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 950].map(
		(n) => `var(--neutral-${n})`
	);
	const status = ['success', 'warning', 'danger', 'info'].map((s) => `var(--fill-${s})`);
	const art = ['blue', 'sky', 'navy', 'green', 'amber'].map((a) => `var(--art-${a})`);

	const groups = [
		{ colours: greys, label: 'Greys', note: 'Pages, surfaces, lines and text', width: 6 },
		{ colours: status, label: 'Status', note: 'Status, and nothing else', width: 0.3 },
		{ colours: art, label: 'Art', note: 'Artwork, never red', width: 0.25 },
		{ colours: ['var(--art-cinder)'], label: 'Cinder', note: 'The one accent', width: 1 }
	];
</script>

<div class="neutral">
	<div class="strip rounded-control elevated-control">
		{#each groups as g (g.label)}
			{#each g.colours as c (c)}<span style:background={c} style:flex-grow={g.width}></span>{/each}
		{/each}
	</div>

	<dl class="legend">
		{#each groups as g (g.label)}
			<div>
				<dt>
					<span class="chip">
						{#each g.colours as c (c)}<span style:background={c}></span>{/each}
					</span>
					{g.label}
				</dt>

				<dd>{g.note}</dd>
			</div>
		{/each}
	</dl>

	<div class="badges">
		<Badge variant="success">Ready</Badge>

		<Badge variant="warning">Loading</Badge>

		<Badge variant="danger">Stopped</Badge>

		<Badge variant="info">Update available</Badge>

		<Badge variant="secondary">Q4_K_M</Badge>

		<Badge variant="outline">128k context</Badge>
	</div>
</div>

<style>
	.neutral {
		display: grid;
		gap: 28px;
	}

	.strip,
	.chip {
		display: flex;
		overflow: hidden;
	}

	.strip {
		height: 96px;
	}

	.chip {
		box-shadow: inset 0 0 0 1px var(--line-default);
	}

	.strip span {
		flex-basis: 0;
	}

	.legend {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr));
		gap: 16px;
		margin: 0;
	}

	dt {
		display: flex;
		gap: 8px;
		align-items: center;
		font-size: 14px;
		font-weight: 600;
	}

	.chip {
		width: 24px;
		height: 12px;
		border-radius: 3px;
	}

	.chip span {
		flex: 1;
	}

	dd {
		margin: 4px 0 0;
		color: var(--figure-muted);
		font-size: 13px;
	}

	.badges {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		padding-top: 28px;
		border-top: 1px solid var(--line-default);
	}
</style>
