<!--
	Shared: one server run by one person, used by the people around it (a family, a team, a lab, a neighbourhood), each
	on their own machines. Drawn with the communications motifs: the shared server in the middle, the groups around it
	with a laptop or a workstation each, joined to it by dashed lines. The motifs read their inks from the --p-*
	variables, set here from the theme so they follow light and dark.
-->
<script lang="ts">
	import Laptop from '../../brand/communications/posts/motifs/Laptop.svelte';
	import Server from '../../brand/communications/posts/motifs/Server.svelte';
	import WorkstationLarge from '../../brand/communications/posts/motifs/WorkstationLarge.svelte';
	import WorkstationSmall from '../../brand/communications/posts/motifs/WorkstationSmall.svelte';

	const groups = [
		{ machine: Laptop, name: 'A family', on: 'Laptops and phones', side: 'left' },
		{ machine: WorkstationSmall, name: 'A team', on: 'Desktops at work', side: 'right' },
		{ machine: WorkstationLarge, name: 'A lab', on: 'Workstations and notebooks', side: 'left' },
		{ machine: Laptop, name: 'A neighbourhood', on: 'Whatever people have', side: 'right' }
	];
</script>

<div class="sharing">
	<div class="hub">
		<div class="motif big"><Server /></div>

		<b>One server</b>

		<span class="mono">llama-server</span>
	</div>

	{#each groups as group, i (group.name)}
		<div class={['group card-surface rounded-card', group.side, `g${i}`]}>
			<div class="motif"><group.machine /></div>

			<div class="text">
				<b>{group.name}</b>

				<span>{group.on}</span>
			</div>
		</div>
	{/each}
</div>

<style>
	.sharing {
		--p-base: var(--surface-page);
		--p-ink: var(--fg-default);
		--p-muted: var(--figure-muted);
		--p-inset: color-mix(in oklab, var(--fg-default) 6%, transparent);
		position: relative;
		display: grid;
		grid-template-areas:
			'g0 hub g1'
			'g2 hub g3';
		grid-template-columns: minmax(0, 1fr) minmax(0, 1.1fr) minmax(0, 1fr);
		gap: 12px 40px;
		align-items: center;
	}

	.hub {
		display: grid;
		grid-area: hub;
		justify-items: center;
		gap: 2px;
		padding: 12px;
		border-radius: 16px;
		background: var(--surface-page);
	}

	.g0 {
		grid-area: g0;
	}

	.g1 {
		grid-area: g1;
	}

	.g2 {
		grid-area: g2;
	}

	.g3 {
		grid-area: g3;
	}

	.group {
		position: relative;
		display: grid;
		grid-template-columns: 72px minmax(0, 1fr);
		gap: 10px;
		align-items: center;
		padding: 10px 12px;
	}

	/* the dashed line from each group across the gap to the server */
	.group::after {
		position: absolute;
		top: 50%;
		width: 40px;
		border-top: 2px dashed var(--line-default);
		content: '';
	}

	.left::after {
		left: 100%;
	}

	.right::after {
		right: 100%;
	}

	.motif {
		aspect-ratio: 4 / 3;
	}

	.motif.big {
		width: min(100%, 220px);
	}

	.text {
		display: grid;
		gap: 2px;
	}

	b {
		font-size: 15px;
		font-weight: 600;
	}

	.text span {
		color: var(--figure-muted);
		font-size: 12px;
		line-height: 1.4;
	}

	.mono {
		color: var(--art-cinder);
		font: 500 12px var(--mono);
	}

	@media (width < 720px) {
		.sharing {
			grid-template-areas: 'hub hub' 'g0 g1' 'g2 g3';
			grid-template-columns: repeat(2, minmax(0, 1fr));
			gap: 8px;
			align-items: stretch;
		}

		.group {
			grid-template-columns: minmax(0, 1fr);
			align-content: start;
			justify-items: start;
		}

		.motif {
			width: 72px;
		}

		.group::after {
			display: none;
		}
	}
</style>
