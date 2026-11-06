// A small 3D projection for the motifs: points in a unit space (y up) turned by a yaw and a pitch and drawn with a
// gentle perspective into a 400 × 300 SVG, with their depth, so lines and dots can fade and shrink with distance.

export type Point3 = [number, number, number];

export interface View {
	cx: number;
	cy: number;
	/** the camera's distance; larger is flatter */
	distance: number;
	/** how far the camera looks down onto the scene, in radians */
	pitch: number;
	scale: number;
	/** the turn around the vertical axis, in radians */
	yaw: number;
}

/** a point on the page and its depth: 0 at the centre, positive away from the viewer */
export function project([x, y, z]: Point3, v: View): [number, number, number] {
	const x1 = x * Math.cos(v.yaw) - z * Math.sin(v.yaw);
	const z1 = x * Math.sin(v.yaw) + z * Math.cos(v.yaw);
	// a positive pitch looks down onto the scene: what is higher comes nearer
	const y1 = y * Math.cos(v.pitch) + z1 * Math.sin(v.pitch);
	const z2 = -y * Math.sin(v.pitch) + z1 * Math.cos(v.pitch);
	const f = v.distance / (v.distance + z2);

	return [v.cx + x1 * v.scale * f, v.cy - y1 * v.scale * f, z2];
}

/** how far back a depth is, from 0 (nearest) to 1 (farthest), over a range */
export const far = (depth: number, range = 1.6) =>
	Math.min(1, Math.max(0, (depth + range / 2) / range));

/** a points list for a polyline or polygon */
export const pts = (list: number[][]) =>
	list.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');

/** a repeatable pseudo-random sequence, for scattering points the same way on every render */
export function random(seed: number) {
	let s = seed;

	return () => {
		s = (s * 1103515245 + 12345) % 2147483648;

		return s / 2147483648;
	};
}
