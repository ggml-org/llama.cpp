<!--
	A workstation: a tower with its vents and drive bays beside a monitor, the power light in cinder. A motif for
	posts about bigger models on a desk.
-->
<script lang="ts">
	import './motif.css';
</script>

<svg aria-hidden="true" class="motif-svg" viewBox="0 0 400 300">
	<rect class="ink" height="130" rx="10" width="190" x="30" y="50" />

	<path class="ink" d="M 125 180 v 40 M 85 226 h 80" />

	<line style:stroke-width="6" class="ink" x1="56" x2="168" y1="88" y2="88" />

	<line style:stroke-width="6" class="ink" x1="56" x2="194" y1="112" y2="112" />

	<line style:stroke-width="6" class="ink" x1="56" x2="140" y1="136" y2="136" />

	<rect class="ink" height="220" rx="12" width="110" x="260" y="30" />

	{#each Array(6) as _, i (i)}<line
			style:stroke-width="4"
			class="ink"
			x1="284"
			x2="346"
			y1={70 + i * 14}
			y2={70 + i * 14}
		/>{/each}

	<rect style:stroke-width="4" class="ink" height="18" rx="4" width="62" x="284" y="170" />

	<rect style:stroke-width="4" class="ink" height="18" rx="4" width="62" x="284" y="196" />

	<circle class="accent" cx="315" cy="50" r="7" />
</svg>
