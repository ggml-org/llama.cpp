// Reading the specimen's glyph outlines (SVG paths of M, L, H, V, Q and Z, in the glyph grid's coordinates, y down)
// for the overview's close-ups: their points, the angled cuts at stroke ends and the notches where bowls meet stems.

export type Point = [number, number];
export type Segment = { kind: 'L' | 'Q'; from: Point; to: Point; via?: Point };

/** Every contour of a path as straight and quadratic segments. */
export function segments(d: string): Segment[][] {
	const contours: Segment[][] = [];

	let segs: Segment[] = [];
	let cur: Point = [0, 0];
	let start: Point = [0, 0];

	for (const [, cmd, args] of d.matchAll(/([MLHVQZ])([^MLHVQZ]*)/g)) {
		const n = args.trim()
			? args
					.trim()
					.split(/[\s,]+/)
					.map(Number)
			: [];

		if (cmd === 'M') {
			if (segs.length) contours.push(segs);

			segs = [];
			cur = start = [n[0], n[1]];
		} else if (cmd === 'L') {
			for (let i = 0; i < n.length; i += 2)
				segs.push({ from: cur, kind: 'L', to: (cur = [n[i], n[i + 1]]) });
		} else if (cmd === 'H') {
			segs.push({ from: cur, kind: 'L', to: (cur = [n[0], cur[1]]) });
		} else if (cmd === 'V') {
			segs.push({ from: cur, kind: 'L', to: (cur = [cur[0], n[0]]) });
		} else if (cmd === 'Q') {
			for (let i = 0; i < n.length; i += 4) {
				segs.push({
					from: cur,
					kind: 'Q',
					to: (cur = [n[i + 2], n[i + 3]]),
					via: [n[i], n[i + 1]]
				});
			}
		} else if (cmd === 'Z') {
			if (cur[0] !== start[0] || cur[1] !== start[1])
				segs.push({ from: cur, kind: 'L', to: start });

			contours.push(segs);
			segs = [];
		}
	}

	if (segs.length) contours.push(segs);

	return contours;
}

const angleOf = (a: Point, b: Point) =>
	((((Math.atan2(b[1] - a[1], b[0] - a[0]) * 180) / Math.PI) % 180) + 180) % 180;
const square = (angle: number) => [0, 90, 180].some((a) => Math.abs(angle - a) < 5);

/** The angled cuts at stroke ends: short straight edges on a slant, with their angle from the baseline. */
export function cuts(d: string): { a: Point; angle: number; b: Point }[] {
	return segments(d)
		.flat()
		.filter((s) => s.kind === 'L')
		.map((s) => ({ a: s.from, angle: angleOf(s.from, s.to), b: s.to }))
		.filter(({ a, angle, b }) => !square(angle) && Math.hypot(b[0] - a[0], b[1] - a[1]) > 12)
		.filter(({ a, b }) => Math.hypot(b[0] - a[0], b[1] - a[1]) < 120)
		.map(({ a, angle, b }) => ({ a, angle: Math.round(angle > 90 ? 180 - angle : angle), b }));
}

/** The notches where a bowl, arm or shoulder leaves a straight stem: a long vertical edge meeting a curve at a sharp
 *  angle, directly or through a short step (as the font's build finds them). Returns each notch's vertex. */
export function joins(d: string): Point[] {
	const out: Point[] = [];

	for (const segs of segments(d)) {
		const n = segs.length;
		const stem = (s: Segment) =>
			s.kind === 'L' && Math.abs(s.from[0] - s.to[0]) <= 6 && Math.abs(s.from[1] - s.to[1]) > 80;
		const step = (s: Segment) =>
			s.kind === 'L' && Math.hypot(s.to[0] - s.from[0], s.to[1] - s.from[1]) < 30;
		const unit = (a: Point, b: Point): Point => {
			const l = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;

			return [(b[0] - a[0]) / l, (b[1] - a[1]) / l];
		};

		for (let k = 0; k < n; k++) {
			const a = segs[k];
			const b = segs[(k + 1) % n];
			const c = segs[(k + 2) % n];

			let along: Point | undefined, tangent: Point | undefined, vertex: Point | undefined;

			if (stem(a) && b.kind === 'Q')
				[vertex, along, tangent] = [a.to, unit(a.to, a.from), unit(b.from, b.via!)];
			else if (a.kind === 'Q' && stem(b))
				[vertex, along, tangent] = [b.from, unit(b.from, b.to), unit(a.to, a.via!)];
			else if (stem(a) && step(b) && c.kind === 'Q')
				[vertex, along, tangent] = [a.to, unit(a.to, a.from), unit(c.from, c.via!)];
			else if (a.kind === 'Q' && step(b) && stem(c))
				[vertex, along, tangent] = [c.from, unit(c.from, c.to), unit(a.to, a.via!)];

			if (vertex && along && tangent) {
				const cos = along[0] * tangent[0] + along[1] * tangent[1];

				if ((Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI < 60) out.push(vertex);
			}
		}
	}

	return out;
}

/** On-curve points and curve handles, for drawing an outline's anatomy. */
export function points(d: string): { handles: [Point, Point][]; nodes: Point[] } {
	const nodes: Point[] = [];
	const handles: [Point, Point][] = [];

	for (const s of segments(d).flat()) {
		nodes.push(s.to);

		if (s.kind === 'Q') handles.push([s.from, s.via!], [s.via!, s.to]);
	}

	return { handles, nodes };
}
