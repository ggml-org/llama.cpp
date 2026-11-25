// Pan and zoom for the diagram preview. The diagram sits at the top left of its viewport
// with `transform-origin: 0 0`, so a point (x, y) of the diagram shows on screen at
// (x * scale + x0, y * scale + y0), and every zoom keeps one screen point fixed.

export interface DiagramView {
	scale: number;
	x: number;
	y: number;
}

export interface Size {
	width: number;
	height: number;
}

export const DIAGRAM_MIN_SCALE = 0.1;
export const DIAGRAM_MAX_SCALE = 10;
// a fitted diagram may grow, but not so much that a two-node flowchart fills the screen
export const DIAGRAM_MAX_FIT_SCALE = 2;
// one click of a zoom button, or one key press
export const DIAGRAM_ZOOM_FACTOR = 1.25;

function clampScale(scale: number): number {
	return Math.min(Math.max(scale, DIAGRAM_MIN_SCALE), DIAGRAM_MAX_SCALE);
}

/**
 * Scales the view by `factor` around the viewport point (`px`, `py`): what is under that
 * point stays there. The scale is clamped, and the point stays fixed at the clamped scale.
 */
export function zoomAt(view: DiagramView, factor: number, px: number, py: number): DiagramView {
	const scale = clampScale(view.scale * factor);
	const ratio = scale / view.scale;

	return { scale, x: px - (px - view.x) * ratio, y: py - (py - view.y) * ratio };
}

/** Scales the view by `factor` around the middle of the viewport. */
export function zoomAtCenter(view: DiagramView, factor: number, viewport: Size): DiagramView {
	return zoomAt(view, factor, viewport.width / 2, viewport.height / 2);
}

/**
 * The view that shows the whole diagram centered in the viewport, less `padding` on every
 * side. A small diagram grows up to DIAGRAM_MAX_FIT_SCALE.
 */
export function fitView(content: Size, viewport: Size, padding = 0): DiagramView {
	if (content.width <= 0 || content.height <= 0) return { scale: 1, x: 0, y: 0 };

	const scale = clampScale(
		Math.min(
			(viewport.width - 2 * padding) / content.width,
			(viewport.height - 2 * padding) / content.height,
			DIAGRAM_MAX_FIT_SCALE
		)
	);

	return {
		scale,
		x: (viewport.width - content.width * scale) / 2,
		y: (viewport.height - content.height * scale) / 2
	};
}

/**
 * How much a wheel event zooms by. A trackpad pinch arrives as a wheel event with ctrlKey
 * and small deltas, so it is scaled up to feel like the fingers; line and page deltas
 * (mouse wheels on some systems) are turned into pixels first.
 */
export function wheelZoomFactor(event: Pick<WheelEvent, 'ctrlKey' | 'deltaMode' | 'deltaY'>) {
	const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 400 : 1;
	const speed = event.ctrlKey ? 0.01 : 0.002;

	return Math.exp(-event.deltaY * unit * speed);
}
