<!--
	Gradient descent: a loss curve, and the steps a model takes down it, each a dot, the last one at the minimum in
	cinder. A motif for posts about training and fine-tuning.
-->
<script lang="ts">
	import './motif.css';

	const loss = (x: number) => 40 + 0.0042 * (x - 250) ** 2 + 8 * Math.sin(x / 22);
	const curve = Array.from({ length: 81 }, (_, i) => {
		const x = 20 + i * 4.5;

		return `${i ? 'L' : 'M'}${x} ${260 - loss(x) * 0.95}`;
	}).join(' ');
	const steps = [50, 112, 160, 196, 222, 240];
	const at = (x: number) => [x, 260 - loss(x) * 0.95];
</script>

<svg aria-hidden="true" class="motif-svg" viewBox="0 0 400 300">
	<line class="thin" x1="20" x2="380" y1="270" y2="270" />

	<path class="ink" d={curve} />

	{#each steps.slice(0, -1) as s, i (s)}
		{@const [ax, ay] = at(s)}
		{@const [bx, by] = at(steps[i + 1])}
		<line style:opacity="0.6" class="thin" x1={ax} x2={bx} y1={ay} y2={by} />
	{/each}

	{#each steps as s, i (s)}
		{@const [cx, cy] = at(s)}
		<circle
			class={i === steps.length - 1 ? 'accent' : 'solid'}
			{cx}
			{cy}
			r={i === steps.length - 1 ? 13 : 9}
		/>
	{/each}

	<text x="20" y="295">loss</text>
</svg>
