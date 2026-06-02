<script lang="ts">
	import { Download } from '@lucide/svelte';
	import ZoomInIcon from '@lucide/svelte/icons/zoom-in';
	import ZoomOutIcon from '@lucide/svelte/icons/zoom-out';
	import RotateCcwIcon from '@lucide/svelte/icons/rotate-ccw';

	interface Props {
		svgHtml: string;
	}

	let { svgHtml }: Props = $props();

	// Zoom and pan state
	let scale = $state(1);
	let translateX = $state(0);
	let translateY = $state(0);
	let isDragging = $state(false);
	const containerRef = { current: null as HTMLDivElement | null };

	// Drag start position
	let dragStartX = 0;
	let dragStartY = 0;
	let dragStartTranslateX = 0;
	let dragStartTranslateY = 0;

	const MIN_SCALE = 0.1;
	const MAX_SCALE = 10;
	const ZOOM_STEP = 0.15;

	function resetView() {
		scale = 1;
		translateX = 0;
		translateY = 0;
	}

	function zoomIn() {
		scale = Math.min(scale + ZOOM_STEP, MAX_SCALE);
	}

	function zoomOut() {
		scale = Math.max(scale - ZOOM_STEP, MIN_SCALE);
	}

	function handleWheel(event: WheelEvent) {
		event.preventDefault();

		const delta = event.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
		scale = Math.min(Math.max(scale + delta, MIN_SCALE), MAX_SCALE);
	}

	// Imperatively attach a non-passive wheel listener so preventDefault() actually works
	// (Svelte 5 wheel listeners are passive by default, making preventDefault() a no-op)
	$effect(() => {
		const el = containerRef.current;
		if (!el) return;

		function onWheel(e: WheelEvent) {
			handleWheel(e);
		}

		el.addEventListener('wheel', onWheel, { passive: false });
		return () => el.removeEventListener('wheel', onWheel);
	});

	function handlePointerDown(event: PointerEvent) {
		if (event.button !== 0 && event.pointerType === 'mouse') return;

		isDragging = true;
		dragStartX = event.clientX;
		dragStartY = event.clientY;
		dragStartTranslateX = translateX;
		dragStartTranslateY = translateY;

		(event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
	}

	function handlePointerMove(event: PointerEvent) {
		if (!isDragging) return;

		translateX = dragStartTranslateX + (event.clientX - dragStartX);
		translateY = dragStartTranslateY + (event.clientY - dragStartY);
	}

	function handlePointerUp() {
		isDragging = false;
	}
</script>

<div
	bind:this={containerRef.current}
	class="mermaid-preview relative flex items-center justify-center overflow-hidden bg-muted/20"
>
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		class="mermaid-preview-diagram transform-origin-center inline-block min-h-fit min-w-fit will-change-transform {isDragging &&
			'select-none'}"
		style="transform: translate({translateX}px, {translateY}px) scale({scale}); cursor: {isDragging
			? 'grabbing'
			: 'grab'};"
		onpointerdown={handlePointerDown}
		onpointermove={handlePointerMove}
		onpointerup={handlePointerUp}
		onpointerleave={handlePointerUp}
	>
		<!-- eslint-disable-next-line no-at-html-tags -->
		{@html svgHtml}
	</div>

	<div
		class="mermaid-preview-controls absolute bottom-8 flex shrink-0 items-center justify-center p-3"
	>
		<div class="mermaid-preview-controls-inner flex items-center gap-1 rounded-lg bg-muted p-1">
			<button
				class="mermaid-preview-btn flex h-8 w-8 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-foreground transition-colors hover:bg-muted-foreground/15 active:bg-muted-foreground/25"
				onclick={zoomOut}
				title="Zoom out"
				aria-label="Zoom out"
			>
				<ZoomOutIcon class="mermaid-preview-btn-icon h-4 w-4" />
			</button>
			<span
				class="mermaid-preview-zoom-label min-w-[3.5rem] px-0.5 text-center text-xs font-medium text-muted-foreground tabular-nums select-none"
				>{Math.round(scale * 100)}%</span
			>
			<button
				class="mermaid-preview-btn flex h-8 w-8 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-foreground transition-colors hover:bg-muted-foreground/15 active:bg-muted-foreground/25"
				onclick={zoomIn}
				title="Zoom in"
				aria-label="Zoom in"
			>
				<ZoomInIcon class="mermaid-preview-btn-icon h-4 w-4" />
			</button>
			<div class="mermaid-preview-controls-separator mx-1 h-5 w-px bg-border/50" />

			<button
				class="mermaid-preview-btn flex h-8 w-8 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-foreground transition-colors hover:bg-muted-foreground/15 active:bg-muted-foreground/25"
				onclick={resetView}
				title="Reset view"
				aria-label="Reset view"
			>
				<RotateCcwIcon class="mermaid-preview-btn-icon h-4 w-4" />
			</button>
			<div class="mermaid-preview-controls-separator mx-1 h-5 w-px bg-border/50" />

			<button
				class="mermaid-preview-btn flex h-8 w-8 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-foreground transition-colors hover:bg-muted-foreground/15 active:bg-muted-foreground/25"
				onclick={() => {
					if (!svgHtml) return;
					const blob = new Blob([svgHtml], { type: 'image/svg+xml' });
					const url = URL.createObjectURL(blob);
					const a = document.createElement('a');
					a.href = url;
					a.download = 'diagram.svg';
					a.click();
					URL.revokeObjectURL(url);
				}}
				title="Download SVG"
				aria-label="Download SVG"
			>
				<Download class="mermaid-preview-btn-icon h-4 w-4" />
			</button>
		</div>
	</div>
</div>

<style lang="postcss" scoped>
	/* Styles for SVGs rendered via {@html} — no Tailwind class can target child elements */
	.mermaid-preview-diagram :global(svg) {
		min-height: min(50vh, 12rem);
		min-width: min(80vw, 20rem);
		max-width: none !important;
		max-height: none !important;
		height: auto !important;
		width: auto !important;
		display: block;
	}
</style>
