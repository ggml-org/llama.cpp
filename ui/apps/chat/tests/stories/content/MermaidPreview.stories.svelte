<script lang="ts" module>
	import {
		FLOWCHART_SVG,
		PIPELINE_SVG,
		SEQUENCE_SVG,
		SMALL_SVG
	} from '../fixtures/mermaid-preview.js';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { MermaidPreview } from '#lib/components/app/index.ts';
	import { expect, userEvent, waitFor } from 'storybook/test';

	// the pan and zoom viewer of the diagram preview dialog, here in a box of its own
	const { Story } = defineMeta({
		component: MermaidPreview,
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Llama Chat/Content/MermaidPreview'
	});

	// the diagram's transform: where its top left corner is, and its scale
	function readView(canvas: HTMLElement) {
		const diagram = canvas.querySelector<HTMLElement>('.mermaid-preview-diagram')!;
		const m = new DOMMatrix(getComputedStyle(diagram).transform);

		return { scale: m.a, x: m.e, y: m.f };
	}

	// which point of the diagram is under the viewport point (x, y), and where it is now
	function pointUnder(view: ReturnType<typeof readView>, x: number, y: number) {
		return { x: (x - view.x) / view.scale, y: (y - view.y) / view.scale };
	}

	function screenOf(view: ReturnType<typeof readView>, p: { x: number; y: number }) {
		return { x: p.x * view.scale + view.x, y: p.y * view.scale + view.y };
	}
</script>

<Story
	asChild
	name="Flowchart"
	play={async ({ canvasElement }) => {
		const viewport = await waitFor(() => {
			const el = canvasElement.querySelector<HTMLElement>('.mermaid-preview');

			expect(el?.querySelector('.mermaid-preview-diagram')).toBeVisible();

			return el!;
		});
		const rect = viewport.getBoundingClientRect();
		const middle = { x: rect.width / 2, y: rect.height / 2 };
		const fitted = readView(canvasElement);
		// the wheel zooms around the pointer
		const pointer = { x: rect.width * 0.7, y: rect.height * 0.35 };
		const underPointer = pointUnder(fitted, pointer.x, pointer.y);

		viewport.dispatchEvent(
			new WheelEvent('wheel', {
				bubbles: true,
				cancelable: true,
				clientX: rect.left + pointer.x,
				clientY: rect.top + pointer.y,
				deltaY: -200
			})
		);
		await waitFor(() => expect(readView(canvasElement).scale).toBeGreaterThan(fitted.scale));
		await expect(screenOf(readView(canvasElement), underPointer).x).toBeCloseTo(pointer.x, 0);
		await expect(screenOf(readView(canvasElement), underPointer).y).toBeCloseTo(pointer.y, 0);

		// after a pan, the zoom buttons zoom around the middle of the viewport
		const zoomed = readView(canvasElement);
		const drag = (type: string, x: number, y: number) =>
			viewport.dispatchEvent(
				new PointerEvent(type, {
					bubbles: true,
					button: 0,
					clientX: rect.left + x,
					clientY: rect.top + y,
					pointerId: 1
				})
			);

		drag('pointerdown', 100, 100);
		drag('pointermove', 260, 180);
		drag('pointerup', 260, 180);
		await waitFor(() => expect(readView(canvasElement).x).toBeCloseTo(zoomed.x + 160, 0));

		const panned = readView(canvasElement);
		const underMiddle = pointUnder(panned, middle.x, middle.y);

		await userEvent.click(canvasElement.querySelector<HTMLElement>('[aria-label="Zoom out"]')!);
		await waitFor(() => expect(readView(canvasElement).scale).toBeLessThan(panned.scale));
		await expect(screenOf(readView(canvasElement), underMiddle).x).toBeCloseTo(middle.x, 0);
		await expect(screenOf(readView(canvasElement), underMiddle).y).toBeCloseTo(middle.y, 0);

		// reset fits the diagram again
		await userEvent.click(canvasElement.querySelector<HTMLElement>('[aria-label="Reset view"]')!);
		await waitFor(() => expect(readView(canvasElement)).toEqual(fitted));
		await expect(canvasElement.querySelector('.mermaid-preview-zoom-label')?.textContent).toBe(
			`${Math.round(fitted.scale * 100)}%`
		);
	}}
>
	<div class="h-dvh w-screen">
		<MermaidPreview svgHtml={FLOWCHART_SVG} />
	</div>
</Story>

<Story
	asChild
	name="Wide diagram"
	play={async ({ canvasElement }) => {
		// fitting shrinks a diagram wider than the screen so all of it shows
		await waitFor(() => expect(readView(canvasElement).scale).toBeLessThan(1));
	}}
>
	<div class="h-dvh w-screen">
		<MermaidPreview svgHtml={PIPELINE_SVG} />
	</div>
</Story>

<Story asChild name="Small diagram">
	<div class="h-dvh w-screen">
		<MermaidPreview svgHtml={SMALL_SVG} />
	</div>
</Story>

<Story asChild name="Sequence diagram">
	<div class="h-dvh w-screen">
		<MermaidPreview svgHtml={SEQUENCE_SVG} />
	</div>
</Story>
