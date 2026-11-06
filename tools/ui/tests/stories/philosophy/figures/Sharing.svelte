<!--
	Shared: one server run by one person, used by the people around it (a family, a team, a lab, a
	neighbourhood), each with their own devices. Drawn as a hub; the shared server is the cinder node.
-->
<script lang="ts">
	const groups = [
		{ devices: 3, name: 'A family', x: 90, y: 60 },
		{ devices: 5, name: 'A team', x: 550, y: 60 },
		{ devices: 4, name: 'A lab', x: 90, y: 250 },
		{ devices: 6, name: 'A neighbourhood', x: 550, y: 250 }
	];
	const hub = { x: 320, y: 155 };
</script>

<svg
	aria-label="One shared server and the people who use it"
	class="hub"
	role="img"
	viewBox="0 0 640 310"
>
	{#each groups as g (g.name)}
		<line x1={hub.x} x2={g.x} y1={hub.y} y2={g.y} />
	{/each}

	{#each groups as g (g.name)}
		<g transform="translate({g.x} {g.y})">
			<rect class="node" height="56" rx="12" width="150" x="-75" y="-28" />

			<text class="name" text-anchor="middle" y="-4">{g.name}</text>

			{#each Array(g.devices) as _, i (i)}
				<circle class="device" cx={(i - (g.devices - 1) / 2) * 14} cy="14" r="4" />
			{/each}
		</g>
	{/each}

	<g transform="translate({hub.x} {hub.y})">
		<rect class="server" height="70" rx="14" width="170" x="-85" y="-35" />

		<text class="name light" text-anchor="middle" y="-6">One server</text>

		<text class="mono" text-anchor="middle" y="16">llama-server</text>
	</g>
</svg>

<style>
	.hub {
		display: block;
		width: 100%;
		height: auto;
	}

	line {
		stroke: var(--line-default);
		stroke-width: 2;
		stroke-dasharray: 6 6;
	}

	.node {
		fill: var(--surface-page);
		stroke: var(--line-default);
		stroke-width: 1.5;
	}

	.server {
		fill: var(--art-cinder);
	}

	.name {
		fill: var(--fg-default);
		font: 600 15px var(--font-sans);
	}

	.light {
		fill: #111111;
	}

	.mono {
		fill: #111111;
		font: 500 12px var(--mono);
	}

	.device {
		fill: var(--figure-muted);
	}
</style>
