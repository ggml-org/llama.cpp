import {
	DIAGRAM_MAX_FIT_SCALE,
	DIAGRAM_MAX_SCALE,
	DIAGRAM_MIN_SCALE,
	type DiagramView,
	fitView,
	wheelZoomFactor,
	zoomAt,
	zoomAtCenter
} from '#lib/utils/diagram-viewport.ts';
import { describe, expect, it } from 'vitest';

// where the diagram point (x, y) shows on screen
function onScreen(view: DiagramView, x: number, y: number) {
	return { x: x * view.scale + view.x, y: y * view.scale + view.y };
}

// the diagram point under the screen point (x, y)
function underScreen(view: DiagramView, x: number, y: number) {
	return { x: (x - view.x) / view.scale, y: (y - view.y) / view.scale };
}

describe('zoomAt', () => {
	it('keeps the diagram point under the zoom point in place', () => {
		const view = { scale: 1.5, x: -120, y: 40 };
		const under = underScreen(view, 300, 200);
		const zoomed = zoomAt(view, 2, 300, 200);

		expect(zoomed.scale).toBe(3);
		expect(onScreen(zoomed, under.x, under.y).x).toBeCloseTo(300);
		expect(onScreen(zoomed, under.x, under.y).y).toBeCloseTo(200);
	});

	it('clamps the scale and still keeps the point in place', () => {
		const view = { scale: 8, x: 10, y: 10 };
		const under = underScreen(view, 50, 60);
		const zoomed = zoomAt(view, 4, 50, 60);

		expect(zoomed.scale).toBe(DIAGRAM_MAX_SCALE);
		expect(onScreen(zoomed, under.x, under.y).x).toBeCloseTo(50);
		expect(zoomAt({ scale: 0.2, x: 0, y: 0 }, 0.1, 0, 0).scale).toBe(DIAGRAM_MIN_SCALE);
	});
});

describe('zoomAtCenter', () => {
	it('zooms around the middle of the viewport after a pan, not the diagram origin', () => {
		const viewport = { height: 600, width: 800 };
		// zoomed in and panned away from where the diagram started
		const panned = { scale: 2, x: -500, y: -300 };
		const under = underScreen(panned, 400, 300);
		const out = zoomAtCenter(panned, 0.5, viewport);

		expect(out.scale).toBe(1);
		expect(onScreen(out, under.x, under.y).x).toBeCloseTo(400);
		expect(onScreen(out, under.x, under.y).y).toBeCloseTo(300);
	});

	it('zooming in and out again returns to the same view', () => {
		const viewport = { height: 600, width: 800 };
		const view = { scale: 1.3, x: 25, y: -70 };
		const back = zoomAtCenter(zoomAtCenter(view, 1.25, viewport), 1 / 1.25, viewport);

		expect(back.scale).toBeCloseTo(view.scale);
		expect(back.x).toBeCloseTo(view.x);
		expect(back.y).toBeCloseTo(view.y);
	});
});

describe('fitView', () => {
	it('shrinks a large diagram to fit and centers it', () => {
		const view = fitView({ height: 500, width: 2000 }, { height: 800, width: 1000 }, 50);

		expect(view.scale).toBeCloseTo(0.45);
		expect(view.x).toBeCloseTo(50);
		expect(view.y).toBeCloseTo((800 - 500 * 0.45) / 2);
	});

	it('grows a small diagram no more than the fit cap', () => {
		expect(fitView({ height: 50, width: 100 }, { height: 800, width: 1000 }).scale).toBe(
			DIAGRAM_MAX_FIT_SCALE
		);
	});

	it('leaves an unmeasured diagram at its own size', () => {
		expect(fitView({ height: 0, width: 0 }, { height: 800, width: 1000 })).toEqual({
			scale: 1,
			x: 0,
			y: 0
		});
	});
});

describe('wheelZoomFactor', () => {
	it('zooms in on a wheel up and out on a wheel down', () => {
		expect(wheelZoomFactor({ ctrlKey: false, deltaMode: 0, deltaY: -100 })).toBeGreaterThan(1);
		expect(wheelZoomFactor({ ctrlKey: false, deltaMode: 0, deltaY: 100 })).toBeLessThan(1);
	});

	it('a wheel step up and back down cancel out', () => {
		const up = wheelZoomFactor({ ctrlKey: false, deltaMode: 1, deltaY: -3 });
		const down = wheelZoomFactor({ ctrlKey: false, deltaMode: 1, deltaY: 3 });

		expect(up * down).toBeCloseTo(1);
	});

	it('a trackpad pinch zooms faster per pixel than a wheel', () => {
		expect(wheelZoomFactor({ ctrlKey: true, deltaMode: 0, deltaY: -10 })).toBeGreaterThan(
			wheelZoomFactor({ ctrlKey: false, deltaMode: 0, deltaY: -10 })
		);
	});
});
