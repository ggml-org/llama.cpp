<!--
	A server: four rack units, each with its vents and status lights, one light in cinder for the one serving. A motif
	for posts about llama-server, sharing and the largest models.
-->
<script lang="ts">
	import './motif.css';

	const units = [0, 1, 2, 3];
</script>

<svg aria-hidden="true" class="motif-svg" viewBox="0 0 400 300">
	<line class="ink" x1="70" x2="70" y1="20" y2="280" />

	<line class="ink" x1="330" x2="330" y1="20" y2="280" />

	{#each units as u (u)}
		{@const y = 36 + u * 60}
		<rect class="ink" height="44" rx="8" width="240" x="80" {y} />

		{#each Array(8) as _, i (i)}
			<line
				style:stroke-width="4"
				class="ink"
				x1={100 + i * 14}
				x2={100 + i * 14}
				y1={y + 12}
				y2={y + 32}
			/>
		{/each}

		<circle class="solid" cx="270" cy={y + 22} r="6" />

		<circle class={u === 1 ? 'accent' : 'solid'} cx="294" cy={y + 22} r="6" />
	{/each}
</svg>
