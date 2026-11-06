// Boxes for the machine motifs: a box's three faces the viewer sees (its top, its front and its right side, with the
// camera turned as in `device`), shaded by which way they face, and a way to draw on a face in its own units, so a
// fan, a vent or a key sits flat on the surface it belongs to.
import { type Point3, project, pts, type View } from './project';

/** the camera every machine is drawn with: from the front left, a little above, nearly without perspective */
export const device = (scale: number, cy = 160): View => ({
	cx: 200,
	cy,
	distance: 30,
	pitch: 0.5,
	scale,
	yaw: -0.6
});

export interface Box {
	max: Point3;
	min: Point3;
}

export type FaceName = 'front' | 'right' | 'top';

/** how light each face is: the ink laid over the ground, from the top (lightest) to the right side (darkest) */
export const SHADE: Record<FaceName, number> = { front: 0.14, right: 0.3, top: 0.04 };

/** a point on a face, `u` across and `v` up (on the top, `v` runs away from the front), in world units from its corner */
export function on(b: Box, face: FaceName, u: number, v: number): Point3 {
	const [x0, y0, z0] = b.min;
	const [x1, y1] = b.max;

	if (face === 'front') return [x0 + u, y0 + v, z0];

	if (face === 'right') return [x1, y0 + v, z0 + u];

	return [x0 + u, y1, z0 + v];
}

/** a face's size, across and up */
export function size(b: Box, face: FaceName): [number, number] {
	const [dx, dy, dz] = [b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]];

	return face === 'front' ? [dx, dy] : face === 'right' ? [dz, dy] : [dx, dz];
}

/** a drawing pen for a box under a view: points, lines, rectangles and circles on its faces, as SVG points */
export function pen(b: Box, view: View) {
	const at = (face: FaceName, u: number, v: number) => project(on(b, face, u, v), view);

	return {
		/** a point on a face */
		at: (face: FaceName, u: number, v: number) => at(face, u, v).slice(0, 2) as [number, number],
		/** a circle on a face, as polygon points */
		circle: (face: FaceName, u: number, v: number, r: number, n = 36) =>
			pts(
				Array.from({ length: n }, (_, i) => {
					const a = (i / n) * Math.PI * 2;

					return at(face, u + r * Math.cos(a), v + r * Math.sin(a));
				})
			),
		/** the whole face, as polygon points */
		face: (face: FaceName) => {
			const [w, h] = size(b, face);

			return pts([at(face, 0, 0), at(face, w, 0), at(face, w, h), at(face, 0, h)]);
		},
		/** a line on a face, as polyline points */
		line: (face: FaceName, u0: number, v0: number, u1: number, v1: number) =>
			pts([at(face, u0, v0), at(face, u1, v1)]),
		/** a path of points on a face */
		path: (face: FaceName, list: [number, number][]) => pts(list.map(([u, v]) => at(face, u, v))),
		/** a rectangle on a face */
		rect: (face: FaceName, u: number, v: number, w: number, h: number) =>
			pts([at(face, u, v), at(face, u + w, v), at(face, u + w, v + h), at(face, u, v + h)])
	};
}

/** the faces a box shows, back to front, as polygon points with their shade */
export function faces(b: Box, view: View) {
	const p = pen(b, view);

	return (['top', 'front', 'right'] as FaceName[]).map((name) => ({
		name,
		points: p.face(name),
		shade: SHADE[name]
	}));
}

/** a soft shadow on the floor under a box, a little larger than it */
export function shadow(b: Box, view: View, grow = 0.12) {
	const y = b.min[1];

	return pts(
		(
			[
				[b.min[0] - grow, y, b.min[2] - grow],
				[b.max[0] + grow, y, b.min[2] - grow],
				[b.max[0] + grow, y, b.max[2] + grow],
				[b.min[0] - grow, y, b.max[2] + grow]
			] as Point3[]
		).map((q) => project(q, view))
	);
}
