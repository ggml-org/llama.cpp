#!/usr/bin/env node

/**
 * Writes the token sections of DESIGN.md's front matter (`colors`, `typography` and `rounded`) from the CSS they
 * mirror, so the file agents read cannot drift from the code:
 *
 * - the brand's colours, @ggml-org/llama-brand's css/colors.css,
 * - Llama UI's primitives and semantic tokens, ui/lib/ui/src/lib/styles/tokens/{primitives,semantic}.css,
 * - Llama UI's font stacks, ui/lib/ui/src/lib/styles/theme.css.
 *
 * Everything else in DESIGN.md, the `spacing` and `components` sections and the prose, is written by hand.
 *
 * Usage: node scripts/design-md.js [--check]
 *
 * --check writes nothing and exits with 1 if DESIGN.md is not current, for CI (`npm run check:design`).
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const app = new URL('../', import.meta.url);
const styles = new URL('../../../lib/ui/src/lib/styles/', import.meta.url);
const DESIGN_MD = fileURLToPath(new URL('DESIGN.md', app));
const SOURCES = {
	brand: fileURLToPath(new URL('node_modules/@ggml-org/llama-brand/css/colors.css', app)),
	primitives: fileURLToPath(new URL('tokens/primitives.css', styles)),
	semantic: fileURLToPath(new URL('tokens/semantic.css', styles)),
	theme: fileURLToPath(new URL('theme.css', styles))
};
// the selectors of the themes a semantic token may change in, and the suffix its twin takes here
const THEMES = [{ selector: '.dark', suffix: '-dark' }];
// the type roles, by the @theme names Tailwind turns into text-body, text-label-sm, ...: a size with a line height
const TYPE_ROLE = /^--text-([a-z]+(?:-[a-z]+)*)$/;
// body and caption text takes the weight around it, which is regular unless a parent says otherwise
const DEFAULT_WEIGHT = '400';
// code takes the body-sm role in the mono stack (see `:where(code, pre, kbd, samp)` in theme.css)
const CODE_ROLE = 'body-sm';
// Tailwind's own `rounded-full`, for lamps, avatars and round buttons; the other corners are the scale in theme.css
const ROUNDED_FULL = '9999px';
// the sections this script owns, in the order they appear in the front matter; the next one is written by hand
const SECTIONS = ['colors', 'typography', 'rounded'];
const NEXT_SECTION = 'spacing';

/** @param {string} css @returns {string} */
function stripComments(css) {
	return css.replace(/\/\*[\s\S]*?\*\//g, '');
}

/**
 * The custom properties declared in each top-level block, by the block's selector (or at-rule).
 * @param {string} css
 * @returns {Map<string, Map<string, string>>}
 */
function parseBlocks(css) {
	/** @type {Map<string, Map<string, string>>} */
	const blocks = new Map();
	const source = stripComments(css);

	let depth = 0;
	let start = 0;
	let selector = '';

	for (let i = 0; i < source.length; i++) {
		const char = source[i];

		if (char === '{') {
			if (depth === 0) {
				selector = source.slice(start, i).trim().split(';').pop()?.trim() ?? '';
				start = i + 1;
			}

			depth++;
		} else if (char === '}') {
			depth--;

			if (depth === 0) {
				const declarations = blocks.get(selector) ?? new Map();

				for (const [, name, value] of source
					.slice(start, i)
					.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) {
					declarations.set(name, value.replace(/\s+/g, ' ').trim());
				}

				blocks.set(selector, declarations);
				start = i + 1;
			}
		}
	}

	return blocks;
}

/** @param {string} path @returns {Map<string, Map<string, string>>} */
function readBlocks(path) {
	return parseBlocks(readFileSync(path, 'utf8'));
}

/** @param {string} value */
const isColorLiteral = (value) =>
	/^#[0-9a-f]{3,8}$/i.test(value) ||
	value === 'transparent' ||
	/^(oklch|oklab|rgb|hsl)\([^()]*\)$/.test(value);
/** @param {string} value */
const isColorMix = (value) => /^color-mix\(.*\)$/.test(value) && !/\)\s*,/.test(value);
/** a share a theme sets for a derivation, e.g. `--status-surface-share: 11%` @param {string} value */
const isShare = (value) => /^\d+(?:\.\d+)?%$/.test(value);
/** @param {string} value */
const varName = (value) => /^var\(--([\w-]+)\)$/.exec(value)?.[1];

function build() {
	const brand = readBlocks(SOURCES.brand).get(':root') ?? new Map();
	const primitiveBlocks = readBlocks(SOURCES.primitives);
	const primitives = new Map([...brand, ...(primitiveBlocks.get(':root') ?? [])]);
	const semanticBlocks = readBlocks(SOURCES.semantic);
	const semantic = semanticBlocks.get(':root') ?? new Map();
	const inline = semanticBlocks.get('@theme inline') ?? new Map();
	const theme = readBlocks(SOURCES.theme);
	/** @type {Map<string, string>} name (no dashes) to literal colour */
	const primitiveColors = new Map();

	for (const [name, value] of primitives) {
		if (isColorLiteral(value)) primitiveColors.set(name.slice(2), value);
	}

	/** @param {string} value @param {Map<string, string>} scope */
	const isColor = (value, scope) => {
		if (isColorLiteral(value) || isColorMix(value)) return true;

		const ref = varName(value);

		if (!ref) return false;

		if (primitiveColors.has(ref)) return true;

		const next = scope.get(`--${ref}`);

		return next !== undefined && isColor(next, scope);
	};
	/**
	 * A value with the shares it reads written out, as the theme whose declarations are `scope` sets them, so a
	 * derivation reads as the mix it is: `color-mix(in oklab, var(--fill-danger) 11%, var(--surface-page))`.
	 * @param {string} value
	 * @param {Map<string, string>} [scope]
	 */
	const withShares = (value, scope) =>
		value.replace(/var\((--[\w-]+)\)/g, (reference, name) => {
			const share = scope?.get(name) ?? semantic.get(name) ?? '';

			return isShare(share) ? share : reference;
		});
	const semanticColors = [...semantic].filter(([, value]) => isColor(withShares(value), semantic));
	const semanticNames = new Set(semanticColors.map(([name]) => name.slice(2)));
	/**
	 * A token's value as the front matter writes it: a reference to another token, a literal, or a mix of them. A
	 * primitive faded with transparent (`color-mix(in oklch, var(--neutral-100) 30%, transparent)`) is written as the
	 * colour it is, with its alpha (`oklch(0.97 0 0 / 0.3)`).
	 * @param {string} value
	 * @param {(name: string) => string} ref the front matter name a token reference points at
	 */
	const render = (value, ref) => {
		const faded = /^color-mix\(in oklch, var\(--([\w-]+)\) (\d+(?:\.\d+)?)%, transparent\)$/.exec(
			value
		);
		const base = faded && primitiveColors.get(faded[1]);

		if (faded && base && /^oklch\([^/]*\)$/.test(base)) {
			return `${base.slice(0, -1)} / ${Number(faded[2]) / 100})`;
		}

		return value.replace(/var\(--([\w-]+)\)/g, (_, name) => `{colors.${ref(name)}}`);
	};
	/** @type {[string, string][]} */
	const colors = [['primary', '{colors.action-primary}']];

	for (const [name, value] of primitiveColors) colors.push([name, value]);

	/** @type {Map<string, string>} */
	const base = new Map();

	for (const [name, value] of semanticColors) {
		const rendered = render(withShares(value), (n) => n);

		base.set(name, rendered);
		colors.push([name.slice(2), rendered]);
	}

	// a twin per theme wherever the token resolves differently there, directly or through a token it points at
	for (const { selector, suffix } of THEMES) {
		const overrides = semanticBlocks.get(selector) ?? new Map();
		/** @type {Set<string>} */
		const twins = new Set();
		/** @type {Map<string, string>} */
		const rendered = new Map();
		// twins resolve in dependency order: a reference to a token that has a twin points at the twin
		/** @param {string} name */
		const resolve = (name) => {
			if (rendered.has(name)) return /** @type {string} */ (rendered.get(name));

			const value = overrides.get(name) ?? semantic.get(name) ?? '';

			for (const [, ref] of value.matchAll(/var\(--([\w-]+)\)/g)) {
				if (semanticNames.has(ref)) resolve(`--${ref}`);
			}

			const result = render(withShares(value, overrides), (n) =>
				twins.has(n) ? `${n}${suffix}` : n
			);

			rendered.set(name, result);

			if (result !== base.get(name)) twins.add(name.slice(2));

			return result;
		};

		for (const [name] of semanticColors) resolve(name);

		for (const [name] of semanticColors) {
			if (twins.has(name.slice(2)))
				colors.push([`${name.slice(2)}${suffix}`, rendered.get(name) ?? '']);
		}
	}

	const firstFamily = (/** @type {string | undefined} */ stack) =>
		stack?.split(',')[0].trim().replace(/^'|'$/g, '') ?? 'system-ui';
	const sans = firstFamily(theme.get('@theme inline')?.get('--font-sans'));
	const mono = firstFamily(theme.get(':root')?.get('--font-mono'));
	/** @param {string | undefined} value */
	const length = (value) => {
		const name = value && varName(value);

		return (name && primitives.get(`--${name}`)) ?? value ?? '';
	};
	// a line height is written as the line over the size, `calc(1.25 / 0.875)`: the line, in rem
	/** @param {string | undefined} value */
	const lineHeight = (value) => {
		const ratio = /^calc\(([\d.]+) \/ [\d.]+\)$/.exec(value ?? '');

		return ratio ? `${ratio[1]}rem` : length(value);
	};
	/** @type {[string, Record<string, string>][]} */
	const typography = [];

	for (const [name, value] of inline) {
		const role = TYPE_ROLE.exec(name)?.[1];

		if (!role || !inline.has(`${name}--line-height`)) continue;

		typography.push([
			role,
			{
				fontFamily: sans,
				fontSize: length(value),
				fontWeight: inline.get(`${name}--font-weight`) ?? DEFAULT_WEIGHT,
				lineHeight: lineHeight(inline.get(`${name}--line-height`))
			}
		]);
	}

	const code = typography.find(([role]) => role === CODE_ROLE);

	if (code) typography.push(['code', { ...code[1], fontFamily: mono }]);

	/** @type {[string, string][]} */
	const rounded = [];
	// the corner scale is computed from --radius: `calc(var(--radius) + 4px)`, with 1rem as 16px
	const baseRadius = parseFloat(theme.get(':root')?.get('--radius') ?? '') * 16;

	for (const [name, value] of theme.get('@theme inline') ?? []) {
		const step = /^--radius-([a-z0-9]+)$/.exec(name)?.[1];
		const offset = /^calc\(var\(--radius\) ([+-]) ([\d.]+)px\)$/.exec(value);

		if (!step) continue;

		if (offset) rounded.push([step, `${baseRadius + Number(offset[1] + offset[2])}px`]);
		else if (value === 'var(--radius)') rounded.push([step, `${baseRadius}px`]);
	}

	rounded.push(['full', ROUNDED_FULL]);

	/** @param {string} value */
	const quote = (value) => (/^[\w.-]+$/.test(value) ? value : `'${value.replace(/'/g, "''")}'`);
	const lines = ['colors:'];

	for (const [name, value] of colors) lines.push(`  ${name}: '${value.replace(/'/g, "''")}'`);

	lines.push('typography:');

	for (const [role, style] of typography) {
		lines.push(`  ${role}:`);

		for (const [key, value] of Object.entries(style)) lines.push(`    ${key}: ${quote(value)}`);
	}

	lines.push('rounded:');

	for (const [role, value] of rounded) lines.push(`  ${role}: ${value}`);

	return lines.join('\n') + '\n';
}

const check = process.argv.includes('--check');
const current = readFileSync(DESIGN_MD, 'utf8');
const start = current.indexOf(`\n${SECTIONS[0]}:\n`) + 1;
const end = current.indexOf(`\n${NEXT_SECTION}:\n`) + 1;

if (!current.startsWith('---\n') || start <= 0 || end <= start) {
	console.error(`DESIGN.md: no front matter with ${SECTIONS[0]}: ... ${NEXT_SECTION}: in it`);
	process.exit(1);
}

const next = current.slice(0, start) + build() + current.slice(end);

if (check) {
	if (next !== current) {
		console.error(
			'DESIGN.md is out of date with the tokens: run `npm run design:tokens` in ui/apps/chat and commit it.'
		);
		process.exit(1);
	}

	console.log('DESIGN.md is current.');
} else if (next !== current) {
	writeFileSync(DESIGN_MD, next);
	console.log('DESIGN.md updated.');
} else {
	console.log('DESIGN.md is already current.');
}
