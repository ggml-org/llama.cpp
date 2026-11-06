<!--
	A vector space: two axes, a cloud of points that cluster by meaning, and two vectors from the origin, one in
	cinder, with the angle between them. A motif for posts about embeddings, search and retrieval.
-->
<script lang="ts">
	import './motif.css';

	const o = [60, 250];
	const points = [
		[250, 70],
		[272, 92],
		[238, 100],
		[290, 64],
		[262, 120],
		[150, 190],
		[176, 214],
		[134, 206],
		[318, 180],
		[340, 206],
		[300, 210]
	];
	const a = [268, 90];
	const b = [330, 196];
	// the angle between them, as an arc around the origin
	const r = 60;
	const on = ([x, y]: number[]) => {
		const d = Math.hypot(x - o[0], y - o[1]);

		return [o[0] + ((x - o[0]) / d) * r, o[1] + ((y - o[1]) / d) * r];
	};
	const [bx, by] = on(b);
	const [ax, ay] = on(a);
	const arc = `M ${bx} ${by} A ${r} ${r} 0 0 0 ${ax} ${ay}`;
</script>

<svg aria-hidden="true" class="motif-svg" viewBox="0 0 400 300">
	<line style:opacity="0.5" class="thin" x1={o[0]} x2="380" y1={o[1]} y2={o[1]} />

	<line style:opacity="0.5" class="thin" x1={o[0]} x2={o[0]} y1={o[1]} y2="20" />

	{#each points as [cx, cy] (cx + '-' + cy)}<circle
			style:opacity="0.35"
			class="solid"
			{cx}
			{cy}
			r="7"
		/>{/each}

	<line class="ink" x1={o[0]} x2={b[0]} y1={o[1]} y2={b[1]} />

	<circle class="solid" cx={b[0]} cy={b[1]} r="11" />

	<line class="accent-line" x1={o[0]} x2={a[0]} y1={o[1]} y2={a[1]} />

	<circle class="accent" cx={a[0]} cy={a[1]} r="11" />

	<path style:stroke-width="4" class="ink" d={arc} />

	<text x={o[0] + 74} y={o[1] - 30}>θ</text>
</svg>
