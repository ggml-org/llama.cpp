// The blueprint's construction and the numbers the showcase quotes, all derived from the icon's geometry (logo.ts).
import { angle, corners, edges, extend, type Pt, SIZE, SPLIT } from './logo';

/** The blueprint's frame (x0, y0, x1, y1) and its viewBox. */
export const box: [number, number, number, number] = [-320, -160, 920, 760];
export const vb = `${box[0]} ${box[1]} ${box[2] - box[0]} ${box[3] - box[1]}`;

export const deg = angle(...edges.upperLeft).toFixed(1);
export const lower = SIZE - SPLIT;
export const r = Math.round(corners[0].r);
export const trim = Math.round(corners[0].trim);
export const STEPS = [0, 120, 240, 360, 480, 600]; // five steps of 120 across the drawing

// every part of the blueprint belongs to one idea of the mark; hovering or tapping a part highlights its idea
// (null: plain construction, such as the overall frame, which only fades back with the rest)
export type Feature = 'upper' | 'lower' | 'angle' | 'split' | 'steps' | 'corners' | 'triangles';
export type Tag = Feature | null;

// perpendicular to the 60° edges, through a point
const sh = edges.shared;
const dir: Pt = [sh[1][0] - sh[0][0], sh[1][1] - sh[0][1]];
const perp = (p: Pt): [Pt, Pt] => [p, [p[0] - dir[1], p[1] + dir[0]]];
const vertical = (x: number, f: Tag): [Pt, Pt, Tag] => [[x, 0], [x, 1], f];
const horizontal = (y: number, f: Tag): [Pt, Pt, Tag] => [[0, y], [1, y], f];

export const lines: { a: Pt; b: Pt; f: Tag }[] = (
	[
		...STEPS.map((x) => vertical(x, 'steps')),
		vertical(corners[0].tangentFlat[0], 'corners'),
		vertical(corners[1].tangentFlat[0], 'corners'),
		vertical(466.154, 'upper'),
		horizontal(0, null),
		horizontal(SPLIT, 'split'),
		horizontal(SIZE, null),
		horizontal(corners[0].centre[1], 'corners'),
		horizontal(corners[1].centre[1], 'corners'),
		[...edges.upperLeft, 'angle'],
		[...edges.shared, 'angle'],
		[...edges.lowerRight, 'angle'],
		...[
			[240, SPLIT],
			[0, SPLIT],
			[SIZE, SPLIT],
			[120, SIZE],
			[466.154, 0]
		].map((p) => [...perp(p as Pt), 'angle'] as [Pt, Pt, Tag])
	] as [Pt, Pt, Tag][]
).map(([a, b, f]) => {
	const [p, q] = extend(a, b, box);

	return { a: p, b: q, f };
});

// the ideal triangles inside the L: a grid of equilateral triangles 120 a side, set on the split. With the icon's own
// slope a row is 104 high, so the foot is two rows exactly, and every slanted edge, the split and both of the foot's
// flat edges lie on it; only the upper blade's top edge cuts across. Drawn clipped to the icon.
const row = 60 / (-dir[0] / dir[1]); // 104: where the 60° and 120° lines from neighbouring grid points meet
const slanted = (k: number, s: 1 | -1): [Pt, Pt] => [
	[120 * k, SPLIT],
	[120 * k - s * 60, SPLIT + row]
]; // 60° (s 1) or 120°
const range = (from: number, to: number) =>
	Array.from({ length: to - from + 1 }, (_, i) => from + i);

/** The grid's lines, cut to a box (x0, y0, x1, y1). */
export const triangles = (around: [number, number, number, number]): [Pt, Pt][] =>
	[
		...range(-16, 16).map(
			(n) =>
				[
					[0, SPLIT + n * row],
					[1, SPLIT + n * row]
				] as [Pt, Pt]
		),
		...range(-30, 30).flatMap((k) => [slanted(k, 1), slanted(k, -1)])
	]
		.map(([a, b]) => extend(a, b, around))
		.filter(([a, b]) => a && b);
export const lattice = triangles([-20, -20, SIZE + 20, SIZE + 20]);

// blueprint dimensions: two end points, a tick offset, where the label sits
export type Dim = { a: Pt; b: Pt; t: Pt; at: Pt; label: string; f: Tag; rot?: boolean };
const h = (y: number, x0: number, x1: number, label: string, f: Tag): Dim => ({
	a: [x0, y],
	at: [(x0 + x1) / 2, y - 12],
	b: [x1, y],
	f,
	label,
	t: [0, 10]
});
const v = (x: number, y0: number, y1: number, label: string, f: Tag): Dim => ({
	a: [x, y0],
	at: [x - 12, (y0 + y1) / 2],
	b: [x, y1],
	f,
	label,
	rot: true,
	t: [10, 0]
});

export const dims: Dim[] = [
	h(-100, 0, SIZE, `${SIZE}`, null),
	h(-50, 226.15, 466.154, '240', 'upper'),
	v(-100, 0, SPLIT, `${SPLIT}`, 'split'),
	v(-100, SPLIT, SIZE, `${SIZE - SPLIT}`, 'split'),
	v(-160, 0, SIZE, `${SIZE}`, null),
	...STEPS.slice(1).map((x) => h(690, x - 120, x, '120', 'steps')),
	h(740, 240, SIZE, `${SIZE - 240}`, 'lower')
];

export const features: { f: Feature; name: string; note: string }[] = [
	{
		f: 'upper',
		name: 'Upper blade',
		note: `The tall blade, ${SPLIT} high and 240 across: the top arm of the plus.`
	},
	{
		f: 'lower',
		name: 'Lower blade',
		note: `The foot, ${lower} high and 360 across: the plus’s right arm.`
	},
	{
		f: 'angle',
		name: `${Math.round(+deg)}° edges`,
		note: `Every slanted edge runs at ${deg}°, and the cross lines stand square to them.`
	},
	{
		f: 'split',
		name: 'Split',
		note: `${SPLIT} above, ${lower} below: close to two thirds to one third.`
	},
	{
		f: 'steps',
		name: 'Steps of 120',
		note: `Five steps of 120 span the ${SIZE}: the feet start at 0 and 120, the blades meet on 240, the foot’s corner lands on 480.`
	},
	{
		f: 'triangles',
		name: 'Triangles',
		note: `Inside, the L sits on ideal triangles, 120 a side and ${Math.round(row)} high: the slanted edges, the split and the foot all lie on them.`
	},
	{
		f: 'corners',
		name: 'Soft corners',
		note: `Only the two outer corners round off, cut back ${trim} on circles of radius ≈ ${r}.`
	}
];

// where it comes from: undo the lean and the icon is an upright + and L on a 120 grid. The plus already has the
// icon's own arms, mirrored: upright arms 240 wide and 392 long, side arms 208 deep and 360 long. The upper blade
// is its top arm, the foot its right arm, and the centre block they would share is left out.
const lean = 1 / Math.tan(Math.PI / 3); // 30° from upright

export const sheared = `matrix(1 0 ${-lean} 1 ${SIZE * lean} 0)`; // pivots on the baseline, so the feet stay put
export const rect = ([u0, y0, u1, y1]: number[]) => `M${u0} ${y0}H${u1}V${y1}H${u0}Z`;
export const arm = {
	bottom: [-120, SIZE, 120, SIZE + SPLIT],
	centre: [-120, SPLIT, 120, SIZE],
	left: [-480, SPLIT, -120, SIZE],
	right: [120, SPLIT, 480, SIZE],
	top: [-120, 0, 120, SPLIT]
};
export const plus = Object.values(arm).map(rect);
export const parts = [rect(arm.top), rect(arm.right)];

export const specs: [string, string][] = [
	['Drawing', `${SIZE} × ${SIZE}`],
	['Angle', `${deg}° on all four slanted edges`],
	[
		'Split',
		`${SPLIT} : ${lower}  (${Math.round((SPLIT / SIZE) * 100)} : ${Math.round((lower / SIZE) * 100)})`
	],
	['Steps', `five of 120 across ${SIZE}: 0, 120, 240, 360, 480, 600`],
	['Corners', `two 120° outer corners, cut back ${trim}, radius ≈ ${r}`],
	['Colours', '#111111 on light, #FAFAFA on dark'],
	['In Llama Sans', 'U+E000, cap-height tall on the baseline'],
	['Source', 'ggml-org/llama.brand · CC BY-NC 4.0']
];
