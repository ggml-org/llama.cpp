// Geometry of the Llama icon (ggml-org/llama-brand, icon/icon-light.svg, a 600 x 600 drawing).
// Everything the logo page draws is derived from these numbers.

export type Pt = [number, number];

export const SIZE = 600;
export const SPLIT = 392; // where the two blades meet
export const PATHS = [
	'M600 392L504.249 558L504.137 557.929C487.252 584.069 458.193 600 426.864 600H120L240 392H600Z',
	'M240 392H0L199.602 46.0254C216.032 17.5463 246.411 3.25756e-05 279.29 0H466.154L240 392Z'
];

// the four slanted edges, as two points each
export const edges: Record<string, [Pt, Pt]> = {
	lowerRight: [
		[600, 392],
		[504.249, 558]
	],
	shared: [
		[466.154, 0],
		[120, 600]
	], // upper blade's right edge and lower blade's left edge: one line
	upperLeft: [
		[0, 392],
		[199.602, 46.0254]
	]
};

export const angle = (a: Pt, b: Pt) =>
	(Math.atan2(Math.abs(b[1] - a[1]), Math.abs(b[0] - a[0])) * 180) / Math.PI;

// the rounded corners: tangent points, the sharp corner they replace, and the circle they follow
function corner(tangentEdge: Pt, tangentFlat: Pt, sharp: Pt, inward: 1 | -1) {
	const trim = Math.hypot(tangentFlat[0] - sharp[0], tangentFlat[1] - sharp[1]);
	const r = trim * Math.tan(Math.PI / 3); // the corner is 120°, so r = trim · tan 60°

	return {
		centre: [tangentFlat[0], tangentFlat[1] + inward * r] as Pt,
		r,
		sharp,
		tangentEdge,
		tangentFlat,
		trim
	};
}
const xAt = (e: [Pt, Pt], y: number) =>
	e[0][0] + ((y - e[0][1]) * (e[1][0] - e[0][0])) / (e[1][1] - e[0][1]);

export const corners = [
	corner([199.602, 46.0254], [279.29, 0], [xAt(edges.upperLeft, 0), 0], 1),
	corner([504.249, 558], [426.864, 600], [xAt(edges.lowerRight, 600), 600], -1)
];

/** Extend the line through a and b to the edges of a box (x0, y0, x1, y1). */
export function extend(a: Pt, b: Pt, box: [number, number, number, number]): [Pt, Pt] {
	const [x0, y0, x1, y1] = box;
	const dx = b[0] - a[0],
		dy = b[1] - a[1];
	const ts: number[] = [];

	if (dx) ts.push((x0 - a[0]) / dx, (x1 - a[0]) / dx);

	if (dy) ts.push((y0 - a[1]) / dy, (y1 - a[1]) / dy);

	const inside = ts
		.map((t) => [a[0] + dx * t, a[1] + dy * t] as Pt)
		.filter(([x, y]) => x >= x0 - 1e-6 && x <= x1 + 1e-6 && y >= y0 - 1e-6 && y <= y1 + 1e-6);

	inside.sort((p, q) => p[0] - q[0] || p[1] - q[1]);

	return [inside[0], inside.at(-1)!];
}
