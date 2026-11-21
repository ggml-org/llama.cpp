<script lang="ts">
	import MermaidPreviewControls from './MermaidPreviewControls.svelte';
	import { SVG } from '#lib/constants/index.ts';
	import {
		DIAGRAM_ZOOM_FACTOR,
		type DiagramView,
		fitView,
		wheelZoomFactor,
		zoomAt,
		zoomAtCenter
	} from '#lib/utils/diagram-viewport.ts';
	import { mountSvgShadow } from '#lib/utils/svg-shadow.ts';
	import { SvelteMap } from 'svelte/reactivity';

	interface Props {
		svgHtml: string;
	}

	let { svgHtml }: Props = $props();

	// room kept clear around a fitted diagram, and below it for the controls
	const FIT_PADDING_PX = 24;
	const CONTROLS_SPACE_PX = 96;
	// how far an arrow key pans, and with Shift held
	const PAN_STEP_PX = 48;
	const PAN_STEP_LARGE_PX = 192;

	let viewport = $state<HTMLDivElement | null>(null);
	let diagram = $state<HTMLDivElement | null>(null);
	let svgHost = $state<HTMLDivElement | null>(null);

	let view = $state<DiagramView>({ scale: 1, x: 0, y: 0 });
	// Until the user pans or zooms, the diagram keeps fitting the viewport: as the window
	// resizes, and as a streaming svg grows
	let touched = $state(false);
	let fitted = $state(false);

	// Pointers down on the viewport: one pans, two pinch
	const pointers = new SvelteMap<number, { x: number; y: number }>();
	let isDragging = $state(false);

	// Re-mount on every svgHtml change so a live streaming svg keeps rendering while zoomed
	$effect(() => {
		if (svgHost) mountSvgShadow(svgHost, svgHtml, SVG.DIALOG_SHADOW_STYLE);
	});

	function viewportSize() {
		return { height: viewport?.clientHeight ?? 0, width: viewport?.clientWidth ?? 0 };
	}

	function fit() {
		if (!viewport || !diagram) return;

		const { height, width } = viewportSize();
		const content = { height: diagram.offsetHeight, width: diagram.offsetWidth };

		if (!width || !height || !content.width || !content.height) return;

		view = fitView(
			content,
			{ height: Math.max(height - CONTROLS_SPACE_PX, 0), width },
			FIT_PADDING_PX
		);
		fitted = true;
	}

	$effect(() => {
		if (!viewport || !diagram) return;

		const observer = new ResizeObserver(() => {
			if (!touched) fit();
		});

		observer.observe(viewport);
		observer.observe(diagram);

		return () => observer.disconnect();
	});

	function setView(next: DiagramView) {
		view = next;
		touched = true;
	}

	function resetView() {
		touched = false;
		fit();
	}

	function zoomBy(factor: number) {
		setView(zoomAtCenter(view, factor, viewportSize()));
	}

	// a point of the page in the viewport's own coordinates
	function localPoint(clientX: number, clientY: number) {
		const rect = viewport!.getBoundingClientRect();

		return { x: clientX - rect.left, y: clientY - rect.top };
	}

	function isOnControls(target: EventTarget | null) {
		return target instanceof Element && Boolean(target.closest('.mermaid-preview-controls'));
	}

	// Attached imperatively: Svelte's wheel listeners are passive, so preventDefault() would
	// not stop the page (or a pinch, the browser) from zooming
	$effect(() => {
		const el = viewport;

		if (!el) return;

		function onWheel(event: WheelEvent) {
			if (isOnControls(event.target)) return;

			event.preventDefault();

			const point = localPoint(event.clientX, event.clientY);

			setView(zoomAt(view, wheelZoomFactor(event), point.x, point.y));
		}

		el.addEventListener('wheel', onWheel, { passive: false });

		return () => el.removeEventListener('wheel', onWheel);
	});

	function handlePointerDown(event: PointerEvent) {
		if (event.pointerType === 'mouse' && event.button !== 0) return;

		if (isOnControls(event.target)) return;

		pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
		isDragging = true;
		viewport?.setPointerCapture(event.pointerId);
	}

	function handlePointerMove(event: PointerEvent) {
		const last = pointers.get(event.pointerId);

		if (!last) return;

		if (pointers.size === 1) {
			pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
			setView({
				...view,
				x: view.x + event.clientX - last.x,
				y: view.y + event.clientY - last.y
			});

			return;
		}

		// Two fingers: zoom around their midpoint by how much they spread, and move with it
		const [a, b] = [...pointers.values()];
		const before = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
		const spreadBefore = Math.hypot(a.x - b.x, a.y - b.y);

		pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });

		const [c, d] = [...pointers.values()];
		const after = { x: (c.x + d.x) / 2, y: (c.y + d.y) / 2 };
		const spreadAfter = Math.hypot(c.x - d.x, c.y - d.y);

		if (!spreadBefore) return;

		const point = localPoint(before.x, before.y);
		const zoomed = zoomAt(view, spreadAfter / spreadBefore, point.x, point.y);

		setView({ ...zoomed, x: zoomed.x + after.x - before.x, y: zoomed.y + after.y - before.y });
	}

	function handlePointerUp(event: PointerEvent) {
		pointers.delete(event.pointerId);
		isDragging = pointers.size > 0;
	}

	function handleDoubleClick(event: MouseEvent) {
		if (isOnControls(event.target)) return;

		const point = localPoint(event.clientX, event.clientY);

		setView(zoomAt(view, 2, point.x, point.y));
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.ctrlKey || event.metaKey || event.altKey) return;

		const step = event.shiftKey ? PAN_STEP_LARGE_PX : PAN_STEP_PX;

		switch (event.key) {
			case '+':
			case '=':
				zoomBy(DIAGRAM_ZOOM_FACTOR);

				break;
			case '-':
			case '_':
				zoomBy(1 / DIAGRAM_ZOOM_FACTOR);

				break;
			case '0':
				resetView();

				break;
			case 'ArrowLeft':
				setView({ ...view, x: view.x + step });

				break;
			case 'ArrowRight':
				setView({ ...view, x: view.x - step });

				break;
			case 'ArrowUp':
				setView({ ...view, y: view.y + step });

				break;
			case 'ArrowDown':
				setView({ ...view, y: view.y - step });

				break;
			default:
				return;
		}

		event.preventDefault();
	}
</script>

<svelte:window onkeydown={handleKeydown} />

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
	bind:this={viewport}
	class="mermaid-preview relative size-full touch-none overflow-hidden bg-muted/20 select-none {isDragging
		? 'cursor-grabbing'
		: 'cursor-grab'}"
	ondblclick={handleDoubleClick}
	onlostpointercapture={handlePointerUp}
	onpointercancel={handlePointerUp}
	onpointerdown={handlePointerDown}
	onpointermove={handlePointerMove}
	onpointerup={handlePointerUp}
>
	<div
		bind:this={diagram}
		class="mermaid-preview-diagram absolute top-0 left-0 w-max origin-top-left will-change-transform {fitted
			? ''
			: 'invisible'}"
		style="transform: translate({view.x}px, {view.y}px) scale({view.scale});"
	>
		<div bind:this={svgHost}></div>
	</div>

	<MermaidPreviewControls
		onResetView={resetView}
		onZoomIn={() => zoomBy(DIAGRAM_ZOOM_FACTOR)}
		onZoomOut={() => zoomBy(1 / DIAGRAM_ZOOM_FACTOR)}
		scale={view.scale}
		{svgHtml}
	/>
</div>
