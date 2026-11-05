// The origin animation's timeline: from the + of C++ to the Llama icon, then into the blueprint.
// Everything is in the icon's own coordinates (the 600 x 600 drawing), so it shares the blueprint's frame.
import { SIZE, SPLIT } from './logo';

export const clamp = (x: number) => Math.min(1, Math.max(0, x));
export const lerp = (a: number, b: number, p: number) => a + (b - a) * p;

// arms retract and lines draw with a quick start and a long settle; shape changes accelerate and settle
export const outQuart = (x: number) => 1 - (1 - x) ** 4;
export const outCubic = (x: number) => 1 - (1 - x) ** 3;
export const inOutCubic = (x: number) => (x < 0.5 ? 4 * x ** 3 : 1 - (-2 * x + 2) ** 3 / 2);
export const inOutQuart = (x: number) => (x < 0.5 ? 8 * x ** 4 : 1 - (-2 * x + 2) ** 4 / 2);
export const inOutQuint = (x: number) => (x < 0.5 ? 16 * x ** 5 : 1 - (-2 * x + 2) ** 5 / 2);
const linear = (x: number) => x;

// one step at a time, each with a short hold after it, so every stage of the shape gets its moment (seconds);
// every change starts softly, moves through the middle and settles, the bigger moves on the stronger curve
export const steps = [
	{ ease: outCubic, from: 0, name: 'llama.cpp', note: 'It begins with a name: llama.cpp.', to: 0 },
	{ ease: inOutQuart, from: 0.5, name: '.cpp', note: 'Look closer: its .cpp.', to: 1.1 },
	{
		ease: inOutCubic,
		from: 1.25,
		name: 'C++',
		note: 'Its .cpp is C++, the language llama.cpp is written in.',
		to: 1.6
	},
	{
		ease: inOutQuint,
		from: 1.8,
		name: 'Plus',
		note: 'Its plus grows the icon’s own arms: 240 wide upright, 208 deep across.',
		to: 2.4
	},
	// the L: what the plus doesn't need fades out while the camera moves to where the bolt will strike (see Morph and camera)
	{
		ease: linear,
		from: 2.75,
		name: 'L',
		note: 'Keep the top and right arms and leave the centre empty: an L inside the +, stem and foot touching at one point.',
		to: 3.85
	},
	{
		ease: linear,
		from: 4.2,
		name: 'Bolt',
		note: 'Then the idea: a lightning bolt, for speed, which is what llama.cpp is about.',
		to: 5.5
	},
	// the lean isn't a move: the L snaps to 30° in the bolt's flash (see Morph); this step only carries its caption
	{
		ease: linear,
		from: 5.55,
		name: 'Lean',
		note: 'In a flash the bolt leans the L 30°: every edge runs at 60°, with the bolt inside.',
		to: 5.57
	},
	// the grid the lean puts the L on draws out from the icon across the frame, and stays faint until the fill
	{
		ease: linear,
		from: 5.95,
		name: 'Grid',
		note: 'Every edge now lies on a grid of ideal triangles, 120 a side: the L is built on 60°.',
		to: 6.85
	},
	// two wheels roll into the outer corners, then the corners round onto them (see Morph)
	{
		ease: linear,
		from: 7.15,
		name: 'Soften',
		note: 'Two wheels turn into the outer corners, and only those two corners round onto them.',
		to: 8.45
	},
	{ ease: inOutCubic, from: 8.75, name: 'Fill', note: 'Fill it in: the Llama icon.', to: 9.25 },
	{ ease: linear, from: 9.55, name: 'Blueprint', note: 'And this is how it is built.', to: 10.8 }
];
export const END = 10.95;
/** A step's index by name, so the code that times things off a step doesn't depend on the order. */
export const STEP = Object.fromEntries(steps.map((s, i) => [s.name, i])) as Record<string, number>;
export const BLUEPRINT_STEP = STEP.Blueprint;
/** The steps that have a tile under the animation, in order; the bolt is shown inside the lean. */
export const TILES = ['llama.cpp', 'Plus', 'L', 'Lean', 'Soften'];
/** When each part of the blueprint fades in, in seconds after the Blueprint step starts. */
export const BLUEPRINT = {
	dims: [0.55, 1.05],
	grid: [0.1, 0.6],
	lines: [0.3, 0.85],
	parts: [0.25, 0.75],
	title: [0.8, 1.25],
	triangles: [0.45, 1.2]
} as const;

/** How far step i has got at time t, eased (0 before it starts, 1 once it is done). */
export function progress(t: number, i: number) {
	const { ease, from, to } = steps[i];

	return to <= from ? 1 : ease(clamp((t - from) / (to - from)));
}
/** An eased 0 to 1 over a window of seconds, for the parts of the blueprint that fade in one after another. */
export const span = (t: number, from: number, to: number) =>
	inOutCubic(clamp((t - from) / (to - from)));
export const stepAt = (t: number) =>
	Math.max(
		0,
		steps.findLastIndex((s) => t >= s.from)
	);

// "c++" set in Llama Sans Display Bold (font units, y up), placed so its middle + is centred on the plus; the c stays
// lowercase, so the c of "llama.cpp" is simply the c of "c++".
// Its + is two 114-unit bars, so it splits into the same five blocks as the plus, and each block grows into its own.
export const GLYPH = {
	advance: { C: 563, plus: 542 },
	C: 'M293 -12Q212 -12 152 23Q108 48 80 86Q52 125 38 172Q25 219 25 268Q25 352 58 415Q82 458 118 488Q154 517 199 532Q244 547 293 547Q362 547 417 523Q470 499 504 454Q537 409 546 352L405 334Q397 385 367 411Q337 437 293 437Q234 437 201 392Q169 348 169 268Q169 187 201 142Q234 98 293 98Q337 98 368 124Q398 151 404 205L547 190Q538 132 507 88Q482 53 448 31Q414 9 374 -2Q334 -12 293 -12Z', // the lowercase c: llama.cpp's own c is the c of c++
	plus: 'M214 44V231H27V345H214V532H328V345H515V231H328V44Z'
};
export const S = 1.25; // font units to drawing units
export const MID = GLYPH.advance.C; // the middle + starts where the c ends
export const X = -(MID + 271) * S; // the middle + centred on the plus centre
export const Y = (SIZE + SPLIT) / 2 + 288 * S;

// "llama.cpp" in Llama Sans Display Bold, with the font's own advances and kerning (font units, y up), placed so its
// "c" is the c of "c++": each p turns into a + in place, while "llama." steps aside
export const WORD = {
	glyphs: {
		'.': { bounds: [25, 0, 180, 147], d: 'M25 0V147H180V0Z' },
		a: {
			bounds: [25, -12, 538, 547],
			d: 'M212 -12Q129 -12 77 27Q25 65 25 134Q25 204 69 243Q112 281 200 299L375 339Q375 366 369 384Q363 402 350 416Q326 443 278 443Q233 443 208 423Q180 399 173 353H32Q44 452 113 502Q176 547 278 547Q395 547 455 490Q514 433 514 324V72Q514 51 521 36Q528 20 538 20V0H393V76Q386 64 378 53Q358 30 330 16Q303 1 272 -6Q242 -12 212 -12ZM243 86Q283 86 313 102Q344 119 360 150Q375 180 375 215V248L245 216Q205 208 187 191Q169 174 169 147Q169 118 188 102Q207 86 243 86Z'
		},
		c: {
			bounds: [25, -12, 547, 547],
			d: 'M293 -12Q212 -12 152 23Q108 48 80 86Q52 125 38 172Q25 219 25 268Q25 352 58 415Q82 458 118 488Q154 517 199 532Q244 547 293 547Q362 547 417 523Q470 499 504 454Q537 409 546 352L405 334Q397 385 367 411Q337 437 293 437Q234 437 201 392Q169 348 169 268Q169 187 201 142Q234 98 293 98Q337 98 368 124Q398 151 404 205L547 190Q538 132 507 88Q482 53 448 31Q414 9 374 -2Q334 -12 293 -12Z'
		},
		l: { bounds: [55, 0, 194, 710], d: 'M194 0H55V710H194Z' },
		m: {
			bounds: [55, 0, 815, 547],
			d: 'M55 0V535H181L184 446Q192 464 203 480Q225 511 257 529Q289 547 330 547Q398 547 439 508Q469 480 484 438Q493 463 506 482Q528 514 562 530Q597 547 636 547Q692 547 732 524Q763 505 782 476Q800 447 808 412Q815 378 815 344V0H676V307Q676 371 655 403Q634 436 590 436Q561 436 541 420Q523 407 514 388Q505 368 502 346Q498 324 498 304V0H371V304Q371 368 352 402Q333 436 287 436Q258 436 237 420Q220 407 210 388Q201 368 198 346Q194 324 194 304V0Z'
		},
		p: {
			bounds: [55, -150, 577, 547],
			d: 'M55 -150V535H189L191 455Q202 475 217 492Q242 518 276 532Q311 547 350 547Q426 547 476 510Q513 482 535 442Q557 402 567 357Q577 312 577 268Q577 190 551 127Q526 63 475 26Q425 -12 349 -12Q312 -12 278 2Q243 16 218 42Q203 56 194 73V-150ZM314 98Q369 98 401 143Q433 188 433 268Q433 347 401 392Q369 438 314 438Q276 438 250 419Q226 402 214 376Q202 351 198 322Q194 294 194 268Q194 212 208 174Q221 139 248 118Q276 98 314 98Z'
		}
	} as Record<string, { d: string; bounds: number[] }>,
	layout: [
		['l', 0],
		['l', 237],
		['a', 484],
		['m', 1033],
		['a', 1889],
		['.', 2481],
		['c', 2686],
		['p', 3237],
		['p', 3827]
	] as [string, number][]
};
export const WX = X - 2686 * S; // the c of "llama.cpp" is the c of "c++"
/** The letters of ".cpp" that become "c++", and where each one lands (x in "c++", font units). */
export const BECOMES: Record<number, { to: 'C' | 'plus'; x: number }> = {
	6: { to: 'C', x: 0 },
	7: { to: 'plus', x: MID },
	8: { to: 'plus', x: MID + GLYPH.advance.plus }
};
export const BOUNDS = { C: [25, -12, 547, 547], plus: [27, 44, 515, 532] };

// the camera: the word, then the whole plus, then the icon, then out to the blueprint's own frame.
// Each frame is a box to fit (centre x, centre y, width, height), fitted to the stage's shape.
export type Frame = [number, number, number, number];
const name: Frame = [WX + ((55 + 4404) / 2) * S, Y - 280 * S, 4349 * S * 1.15, 860 * S * 1.9]; // "llama.cpp", ink with a margin
const word: Frame = [X + 871 * S, Y - 198 * S, 1693 * S * 1.35, 697 * S * 1.9]; // ".cpp" and "c++" alike, ink with a margin
const plusFrame: Frame = [0, (SIZE + SPLIT) / 2, 1300, 1300]; // the plus runs 0 to SIZE + SPLIT
const iconFrame: Frame = [300, 400, 1300, 1300]; // room below for the lightning's tip and its glow

export function camera(t: number, aspect: number, final: Frame) {
	const fit = ([cx, cy, w, h]: Frame) => [cx, cy, w / h > aspect ? w : h * aspect];
	const frames = [name, word, plusFrame, iconFrame, final].map(fit);
	const over = (from: number, to: number) => inOutQuart(clamp((t - from) / (to - from)));
	const legs = [
		over(steps[STEP['.cpp']].from, steps[STEP['.cpp']].to), // the zoom to .cpp, before any letter changes
		over(steps[STEP.Plus].from, steps[STEP.Plus].to),
		over(steps[STEP.L].from, steps[STEP.L].to), // to where the bolt will strike, while the rest of the plus fades; it stays there
		over(steps[BLUEPRINT_STEP].from, steps[BLUEPRINT_STEP].from + 0.8)
	];

	let [cx, cy, vw] = frames[0];

	legs.forEach((p, i) => ([cx, cy, vw] = [cx, cy, vw].map((v, j) => lerp(v, frames[i + 1][j], p))));
	const vh = vw / aspect;

	return `${cx - vw / 2} ${cy - vh / 2} ${vw} ${vh}`;
}

// the lightning that strikes over the leaned L, after the sketch: the upper blade cut along its diagonal, from its
// bottom-left corner to its top-right one, keeps its lower-right half; the lower half is that same triangle turned
// 180° about the joint. So its top runs along the foot's top edge, its left side is the foot's left 60° edge carried
// on down, its long side runs parallel to the upper diagonal, and it ends in a tip below and left of the foot.
const turn = ([x, y]: [number, number]): [number, number] => [2 * 240 - x, 2 * SPLIT - y]; // 180° about the joint
const upper: [number, number][] = [
	[466.154, 0],
	[0, SPLIT],
	[240, SPLIT]
];
const lower = upper.map(turn);

export const TIP = lower[0]; // (14, 784)
const tri = (pts: [number, number][]) => `M${pts.map((p) => p.join(' ')).join('L')}Z`;

export const LIGHTNING = tri(upper) + tri(lower);
