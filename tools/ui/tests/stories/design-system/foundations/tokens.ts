import components from '../../../../src/lib/styles/tokens/components.css?raw';
import primitives from '../../../../src/lib/styles/tokens/primitives.css?raw';
import semantic from '../../../../src/lib/styles/tokens/semantic.css?raw';

export interface ColorToken {
	name: string;
	/** What the token points at in light mode, as written in the CSS. */
	light: string;
	/** The dark-mode value when it differs, as written in the CSS. */
	dark?: string;
}

export interface TokenGroup {
	title: string;
	tokens: ColorToken[];
}

// colour tokens only: lengths, shadows and blur are documented elsewhere
const COLOR_VALUE =
	/oklch|color-mix|transparent|var\(--(neutral|cool-gray|red|yellow|green|white|black|surface|fg|line|edge|action|field|status|highlight|sheen|user-bubble|fg-danger)/;
const NOT_COLOR = /px|rem|var\(--(length|elevation|blur|radius|font|line-height)/;

function block(css: string, selector: string): string {
	const start = css.indexOf(`${selector} {`);

	if (start === -1) return '';

	return css.slice(start, css.indexOf('\n}', start));
}

function values(css: string): Map<string, string> {
	return new Map([...css.matchAll(/--([a-z0-9-]+):\s*([^;]+);/g)].map((m) => [m[1], m[2].trim()]));
}

/** Groups the colour tokens of one tier by the comments that head each group in its CSS. */
function groups(css: string): TokenGroup[] {
	const dark = values(block(css, '.dark'));
	const out: TokenGroup[] = [];

	for (const line of block(css, ':root').split('\n')) {
		const comment = line.match(/^\t\/\* (.+?)(?::.*)? \*\/$/);
		const token = line.match(/^\t--([a-z0-9-]+):\s*([^;]+);/);

		if (comment) {
			out.push({ title: comment[1], tokens: [] });
		} else if (token && COLOR_VALUE.test(token[2]) && !NOT_COLOR.test(token[2])) {
			if (!out.length) out.push({ title: 'tokens', tokens: [] });

			out[out.length - 1].tokens.push({
				dark: dark.get(token[1]),
				light: token[2],
				name: token[1]
			});
		}
	}

	return out.filter((group) => group.tokens.length);
}

export const primitiveGroups = groups(primitives);
export const semanticGroups = groups(semantic);
export const componentGroups = groups(components);

/** Shortens var(--neutral-950) to neutral-950 and a color-mix to "neutral-900 at 90%". */
export function reference(value: string): string {
	return value
		.replace(/color-mix\(in oklch, var\(--([a-z0-9-]+)\) (\d+)%, transparent\)/g, '$1 at $2%')
		.replace(/var\(--([a-z0-9-]+)\)/g, '$1');
}
