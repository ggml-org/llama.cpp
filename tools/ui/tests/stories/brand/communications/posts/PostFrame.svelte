<!--
	A piece at its real pixel size, scaled down to fit its column. It sets the ground (grounds.css) and the format's
	grid: --u, the unit every size is a multiple of, and an area inset by the margin and the platform's safe area, which
	the layouts fill. `overlay` draws the grid on top: the margins, the columns and, on a story, the safe area.
-->
<script lang="ts">
	import '../../fonts/fonts.css';
	import './grounds.css';
	import { grid, type Ground, type PostFormat } from './post';
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
		format: PostFormat;
		label: string;
		overlay?: boolean;
		tone: Ground;
	}

	let { children, format, label, overlay = false, tone }: Props = $props();

	let column = $state(0);
	const g = $derived(grid(format));
	const safe = $derived(format.safe ?? { bottom: 0, top: 0 });
	const scale = $derived(column ? Math.min(1, column / format.width) : 0);
</script>

<div bind:clientWidth={column} style:height="{format.height * scale}px" class="fit">
	<article
		style:--cols={g.columns}
		style:--gutter="{g.gutter}px"
		style:--m="{g.margin}px"
		style:--safe-bottom="{safe.bottom}px"
		style:--safe-top="{safe.top}px"
		style:--u="{g.unit}px"
		style:height="{format.height}px"
		style:transform="scale({scale})"
		style:width="{format.width}px"
		aria-label={label}
		class="post ground ground-{tone} shape-{g.shape}"
	>
		<div class="area">{@render children()}</div>

		{#if overlay}
			<div aria-hidden="true" class="overlay">
				{#if safe.top}<i class="safe top"></i><i class="safe bottom"></i>{/if}

				<div class="columns">
					{#each Array(g.columns) as _, i (i)}<i></i>{/each}
				</div>
			</div>
		{/if}
	</article>
</div>

<style>
	.fit {
		width: 100%;
		overflow: hidden;
		border-radius: 12px;
		box-shadow: 0 0 0 1px var(--border);
	}
	.post {
		position: relative;
		box-sizing: border-box;
		overflow: hidden;
		transform-origin: 0 0;
	}
	.area {
		position: absolute;
		inset: calc(var(--m) + var(--safe-top)) var(--m) calc(var(--m) + var(--safe-bottom));
		display: flex;
		flex-direction: column;
	}
	.overlay {
		position: absolute;
		inset: 0;
		pointer-events: none;
	}
	.columns {
		position: absolute;
		inset: calc(var(--m) + var(--safe-top)) var(--m) calc(var(--m) + var(--safe-bottom));
		display: grid;
		grid-template-columns: repeat(var(--cols), 1fr);
		gap: var(--gutter);
		outline: 2px solid rgb(255 64 129 / 0.7);
	}
	.columns i {
		background: rgb(255 64 129 / 0.1);
	}
	.safe {
		position: absolute;
		inset-inline: 0;
		background: repeating-linear-gradient(
			-45deg,
			rgb(255 64 129 / 0.16) 0 8px,
			transparent 8px 16px
		);
	}
	.top {
		top: 0;
		height: var(--safe-top);
	}
	.bottom {
		bottom: 0;
		height: var(--safe-bottom);
	}
</style>
