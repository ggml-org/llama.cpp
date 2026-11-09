// The quantized L, the one swag print drawn in code rather than taken from llama-brand. Ported from the Swag Creator
// (huggingface.co/spaces/ggml-org/swag-creator, src/lib/artworks/generative.ts): the same 40 px grid, the same layer
// ladder and the same legend, with two changes for the design system: the legend's accent is cinder (art-cinder,
// #EE6C1E; the app uses #F65E00) and it is set in Llama Mono. The geometry stays as the app draws it.

export type Tone = 'dark' | 'light';

const SIZE = 600;
const PIXEL = 40;
const LEGEND_WIDTH = 900;
const ACCENT = '#EE6C1E';
/** The icon's two pieces, verbatim from llama-brand icon/icon-*.svg */
const STROKE_PATH =
	'M240 392H0L199.602 46.0254C216.032 17.5463 246.411 3.25756e-05 279.29 0H466.154L240 392Z';
const FOOT_PATH =
	'M600 392L504.249 558L504.137 557.929C487.252 584.069 458.193 600 426.864 600H120L240 392H600Z';

type Point = [number, number];

/** polygon approximations of the pieces, for deciding which cells fall inside */
const STROKE: Point[] = [
	[240, 392],
	[0, 392],
	[199.6, 46],
	[216, 17.5],
	[246.4, 0],
	[279.3, 0],
	[466.2, 0]
];
const FOOT: Point[] = [
	[600, 392],
	[504.2, 558],
	[487.3, 584.1],
	[458.2, 600],
	[426.9, 600],
	[120, 600],
	[240, 392]
];

export interface QuantizationLayer {
	label: string;
	/** how many rows of base pixels the layer covers, top to bottom */
	rows: number;
	/** squares per base pixel side, or `smooth` for the true vector outline */
	splits: 'smooth' | number;
}

/** llama.cpp's everyday ladder, F16 → Q8 → Q4 → Q2: the true logo at the top, half the precision each step down */
export const QUANTIZATION_LAYERS: QuantizationLayer[] = [
	{ label: '16-bit', rows: 4, splits: 'smooth' },
	{ label: '8-bit', rows: 4, splits: 4 },
	{ label: '4-bit', rows: 4, splits: 2 },
	{ label: '2-bit', rows: 3, splits: 1 }
];

/** the artwork's size in its own units: the L and the legend beside it, 1.5 times wider than tall */
export const QUANTIZED_SIZE = { height: SIZE, width: LEGEND_WIDTH };

const ink = (tone: Tone) => (tone === 'light' ? '#111111' : '#FAFAFA');

function insidePolygon([x, y]: Point, polygon: Point[]) {
	let inside = false;

	for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
		const [xi, yi] = polygon[i];
		const [xj, yj] = polygon[j];

		if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
	}

	return inside;
}

const insideL = (point: Point) => insidePolygon(point, STROKE) || insidePolygon(point, FOOT);

/**
 * The quantized L's markup, for an <svg> with viewBox 0 0 900 600: `light` has dark ink for light fabric, `dark` light
 * ink for dark fabric. `id` keeps its clip paths apart when several sit on one page.
 */
export function quantized(tone: Tone, id: string, layers = QUANTIZATION_LAYERS) {
	// a grey close to the ink fills the gaps, which keeps the squares apart without a flickering grid
	const gapColor = tone === 'light' ? '#4d4d4d' : '#c4c4c4';
	const guide = tone === 'light' ? '#999999' : '#808080';
	const sections = layers.map((layer, index) => {
		const from = layers.slice(0, index).reduce((sum, { rows }) => sum + rows, 0) * PIXEL;

		return { ...layer, from, to: Math.min(SIZE, from + layer.rows * PIXEL) };
	});
	const dash = `stroke="${guide}" stroke-width="4" stroke-dasharray="14 10"`;
	const lines = sections
		.slice(1)
		.map(({ from }) => `<line x1="0" y1="${from}" x2="${LEGEND_WIDTH}" y2="${from}" ${dash}/>`);
	const labels = sections.map(
		({ from, label, to }) =>
			`<text x="${LEGEND_WIDTH}" y="${(from + to) / 2}" text-anchor="end" dominant-baseline="central" font-family="'Llama Mono', ui-monospace, monospace" font-size="44" font-weight="600" fill="${ACCENT}">${label}</text>`
	);
	const layersMarkup = sections
		.map(({ from, splits, to }, index) => {
			if (splits === 'smooth') {
				const clip = `${id}-layer-${index}`;

				return `<clipPath id="${clip}"><rect y="${from}" width="${SIZE}" height="${to - from}"/></clipPath><g clip-path="url(#${clip})"><path d="${STROKE_PATH}"/><path d="${FOOT_PATH}"/></g>`;
			}

			const cell = PIXEL / splits;
			// finer squares get proportionally thinner gaps, so they don't fade into grey
			const gap = cell >= 10 ? Math.max(1.5, cell / 12) : cell * 0.18;
			const squares: string[] = [];

			for (let y = from; y < to; y += PIXEL) {
				for (let x = 0; x < SIZE; x += PIXEL) {
					if (!insideL([x + PIXEL / 2, y + PIXEL / 2])) continue;

					for (let i = 0; i < splits; i++) {
						for (let j = 0; j < splits; j++) {
							const sx = x + i * cell;
							const sy = y + j * cell;

							squares.push(
								`<rect x="${sx}" y="${sy}" width="${cell}" height="${cell}" fill="${gapColor}"/>`,
								`<rect x="${sx + gap / 2}" y="${sy + gap / 2}" width="${cell - gap}" height="${cell - gap}"/>`
							);
						}
					}
				}
			}

			return squares.join('');
		})
		.join('');

	// lines first, so the pixels sit on top of them
	return `${lines.join('')}<g fill="${ink(tone)}">${layersMarkup}</g>${labels.join('')}`;
}
