#!/usr/bin/env node

/**
 * Exports the Llama design system for Claude (a Design System artifact that Claude Design reads) from this
 * repository and its Storybook, so the Storybook stays the source of truth and the Claude system is refreshed from
 * it after a change:
 *
 *   npm run export:design-system                 # builds Storybook if needed, writes design-system-export/
 *   npm run export:design-system -- --storybook storybook-static --out design-system-export
 *   npm run export:design-system -- index --uploads uploads.json --title "Llama"
 *
 * The export writes, under <out>/project/: README.md (BRAND.md and an index of what is in the system),
 * tokens.json, the fonts, the logos and motifs under assets/ (rendered from the stories with their colours baked in),
 * and a component per key visual with its guidelines, a preview and example images rendered at their real size.
 * Images under assets/, and the example images the previews show (listed in <out>/preview-images.json), are
 * uploaded to the system as assets; the `index` step then writes the system's index (project/design-system.json) and
 * points the previews at their uploads, from the upload results, `{ "<path under project/>": "/_blob/<id>" }`.
 */

import { execSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SPACE = 'https://huggingface.co/spaces/ggml-org/llama-storybook';
const args = process.argv.slice(2);
const flag = (name, fallback) => {
	const i = args.indexOf(`--${name}`);

	return i >= 0 ? args[i + 1] : fallback;
};
const OUT = path.resolve(ROOT, flag('out', 'design-system-export'));
const PROJECT = path.join(OUT, 'project');
const write = (rel, data) => {
	const file = path.join(PROJECT, rel);

	fs.mkdirSync(path.dirname(file), { recursive: true });
	fs.writeFileSync(file, data);
};
const slug = (s) =>
	s
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, '-')
		.replace(/^-|-$/g, '');
// the key visuals, each a component, with the stories its examples come from: the layout's own story and the
// applications built on it (an example's file is named after the post's own label, so both kinds sit together)
const KEY_VISUALS = [
	{
		name: 'Statement',
		shape: [
			'Header: the lockup and the URL.',
			'Display headline (3.2u), two or three lines; a lede (0.72u) under it; a command in Mono if it helps.',
			'A motif fills the rest: beside the text on wide formats (columns 8 to 12), above it on square and tall ones.',
			'Footer: who it is from, the promise.'
		],
		stories: ['communications-key-visuals--statement'],
		summary:
			'Statement says one thing big: a display headline, a lede and a motif filling the rest.',
		use: 'A launch line, a manifesto, the repo’s link preview. One idea, a few words.'
	},
	{
		name: 'Product',
		shape: [
			'The lockup, the display headline and a caption chip (solid) on the left half (columns 1 to 7).',
			'The real app (menu bar and panel) on the right; on square and tall formats it sits under the text and runs off the bottom edge.',
			'On the day sky or the night sky.'
		],
		stories: ['communications-key-visuals--product', 'communications-applications--link-previews'],
		summary: 'Product shows the app: the headline beside the real Llama app in the menu bar.',
		use: 'Link previews of llama.app, launch images, anything about the product itself.'
	},
	{
		name: 'Roster',
		shape: [
			'Header: the icon, a Mono kicker (who it is from), the URL.',
			'Headline (2u) over four columns with a motif beside it; the lede and use cases as outline chips.',
			'A Mono label, then the cards: one across on tall formats, two on square, three on wide; the leading item tagged in cinder.'
		],
		stories: ['communications-key-visuals--roster', 'communications-applications--model-drop'],
		summary:
			'Roster presents several items of one kind: a headline with a motif, chips, then a row of cards.',
		use: 'A model drop, a set of new features, several models at once.'
	},
	{
		name: 'Bulletin',
		shape: [
			'Header: the lockup and a pill with a cinder dot (“New today”).',
			'Headline (2u) with the lede beside it on wide formats.',
			'A lead item (a family card, a chart) and a column of others (cards, stat cards); the footer.'
		],
		stories: ['communications-key-visuals--bulletin', 'communications-applications--what-s-new'],
		summary:
			'Bulletin carries news with several items: a headline and lede, a lead item and a side column.',
		use: 'What’s new, release notes, a monthly roundup.'
	},
	{
		name: 'Numeral',
		shape: [
			'The figure (5u to 9u) in cinder with its Mono label under it, over a dot grid.',
			'A headline and lede that say what it means: beside it on wide formats, under it otherwise.',
			'The figure is the one accent: nothing else in cinder.'
		],
		stories: ['communications-key-visuals--numeral'],
		summary: 'Numeral lets one figure carry the piece: the number huge in cinder over a dot grid.',
		use: 'Speed, size, a count, a milestone: when one real number is the news.'
	}
];
const STORY_PATHS = {
	'communications-applications--link-previews': 'Communications › Applications › Link previews',
	'communications-applications--model-drop': 'Communications › Applications › Model drop',
	'communications-applications--what-s-new': 'Communications › Applications › What’s new',
	'communications-key-visuals--bulletin': 'Communications › Key visuals › Bulletin',
	'communications-key-visuals--numeral': 'Communications › Key visuals › Numeral',
	'communications-key-visuals--product': 'Communications › Key visuals › Product',
	'communications-key-visuals--roster': 'Communications › Key visuals › Roster',
	'communications-key-visuals--statement': 'Communications › Key visuals › Statement'
};
const LOGOS = 'tests/stories/brand/assets/files/logo/svg';
const LOGO_FILES = [
	['icon', 'icon-light.svg'],
	['icon', 'icon-dark.svg'],
	['icon', 'icon-light-bg.svg'],
	['icon', 'icon-dark-bg.svg'],
	['lockup/llama', 'logo-llama-light.svg'],
	['lockup/llama', 'logo-llama-dark.svg'],
	['lockup/llama', 'logo-llama-light-bg.svg'],
	['lockup/llama', 'logo-llama-dark-bg.svg'],
	['lockup/llama-cpp', 'logo-llama-cpp-light.svg'],
	['lockup/llama-cpp', 'logo-llama-cpp-dark.svg'],
	['lockup/llama-cpp', 'logo-llama-cpp-light-bg.svg'],
	['lockup/llama-cpp', 'logo-llama-cpp-dark-bg.svg']
];
// the app icon, for Llama as a thing in a piece: the macOS icon cropped to its tile, named as the other logos are (its
// sources are named for the Mac's mode, so the ink tile for light grounds is the dark one)
const APP_ICONS = [
	['app/apple/macos-1024x1024-dark.svg', 'app-icon-light.svg'],
	['app/apple/macos-1024x1024-light.svg', 'app-icon-dark.svg']
];

function tokens() {
	const primitives = fs.readFileSync(
		path.join(ROOT, 'src/lib/styles/tokens/primitives.css'),
		'utf8'
	);
	const grounds = fs.readFileSync(
		path.join(ROOT, 'tests/stories/brand/communications/posts/grounds.css'),
		'utf8'
	);
	const declared = (css, prefix) =>
		[...css.matchAll(new RegExp(`--(${prefix}[a-z0-9-]*):\\s*([^;]+);`, 'g'))].map((m) => [
			m[1],
			m[2].trim()
		]);
	const block = (name) => {
		const m = new RegExp(`\\.ground-${name} \\{([\\s\\S]*?)\\n\\}`).exec(grounds);

		return Object.fromEntries(
			[...m[1].matchAll(/--p-([a-z-]+):\s*([^;]+);/g)].map((x) => [x[1], x[2].trim()])
		);
	};
	const plain = (v) => /^(#[0-9a-f]{3,8}|rgba?\([^()]*\)|oklch\([^()]*\))$/i.test(v);
	const primitive = Object.fromEntries(declared(primitives, ''));
	// a colour as the tokens can hold it: a plain value, an alias of a primitive, or a primitive mixed with
	// transparent written as that oklch colour with its alpha
	const resolve = (value) => {
		if (plain(value)) return value;

		if (/^var\(--[a-z0-9-]+\)$/.test(value)) return `{${value.slice(6, -1)}}`;

		const mix = /^color-mix\(in oklab, var\(--([a-z0-9-]+)\) (\d+)%, transparent\)$/.exec(value);
		const base = mix && /^oklch\(([^()/]*)\)$/.exec(primitive[mix[1]] ?? '');

		return base ? `oklch(${base[1].trim()} / ${+mix[2] / 100})` : null;
	};
	const color = [];
	const add = (name, value, usage) => color.push({ name, usage, value });

	for (const [name, value] of declared(primitives, 'neutral-'))
		add(name, value, 'A neutral grey. Grounds, inks and hairlines come from these.');
	for (const [name, value] of declared(primitives, 'sand-'))
		add(name, value, 'Sand, the warm neutral: the sand ground (sand-50) and its ink (sand-900).');
	for (const [name, value] of declared(primitives, 'art-'))
		add(
			name,
			value,
			name === 'art-cinder'
				? 'Cinder, the one accent: one thing per piece (the chosen path, the figure that matters, a tag, the .cpp of the logotype). Never a ground.'
				: 'An art colour, for illustration and artwork details. Never a ground for a whole piece, never text.'
		);

	const GROUND_NOTES = {
		dark: 'The dark ground: the developer side, the engine. Takes either lockup.',
		day: 'The day sky, llama.app’s hero, a light blue gradient deeper at the top. The Llama lockup only.',
		light: 'The light ground, the default. Takes either lockup.',
		night: 'The night sky, a deep blue gradient darkest at the top. The Llama lockup only.',
		sand: 'The sand ground: a warm, quiet piece. The Llama lockup only.'
	};
	const ROLES = {
		base: 'flat colour of the ground',
		bg: 'background',
		card: 'card fill',
		ink: 'text and lines',
		inset: 'inset fill (terminals, badges)',
		line: 'hairline',
		link: 'links and repo ids',
		muted: 'muted text (ledes, labels, footers)',
		solid: 'solid chip fill',
		'solid-ink': 'text on the solid chip'
	};

	for (const ground of ['light', 'dark', 'sand', 'day', 'night']) {
		const values = block(ground);

		for (const [role, value] of Object.entries(values)) {
			if (!ROLES[role]) continue;

			const resolved = resolve(value);

			if (resolved)
				add(`${ground}-${role}`, resolved, `${GROUND_NOTES[ground]} Its ${ROLES[role]}.`);
		}
	}

	// the skies are gradients: their stops, top to bottom
	add(
		'day-sky-top',
		'#c4ddf6',
		'The day sky’s top stop (linear-gradient 160deg: #c4ddf6, #d0e4f7, #dfecf9).'
	);
	add('day-sky-middle', '#d0e4f7', 'The day sky’s middle stop.');
	add('day-sky-bottom', '#dfecf9', 'The day sky’s bottom stop, the horizon.');
	add(
		'night-sky-top',
		'#0f1a2b',
		'The night sky’s top stop (linear-gradient 180deg: #0f1a2b, #13213a, #182a45), with a faint glow at the horizon.'
	);
	add('night-sky-middle', '#13213a', 'The night sky’s middle stop.');
	add('night-sky-bottom', '#182a45', 'The night sky’s bottom stop, the horizon.');

	const style = (name, u, size, extra) => ({
		fontSize: `${+(u * size).toFixed(2)}px`,
		name,
		...extra
	});
	const scale = (u, label) => ({
		family: 'sans',
		name: label,
		styles: [
			style('display', u, 3.2, {
				fontWeight: 700,
				letterSpacing: '-0.035em',
				lineHeight: 0.92,
				usage: 'Display, 3.2u: one statement, two or three lines.'
			}),
			style('headline', u, 2, {
				fontWeight: 700,
				letterSpacing: '-0.02em',
				lineHeight: 0.98,
				usage: 'Headline, 2u.'
			}),
			style('figure', u, 5, {
				fontWeight: 700,
				letterSpacing: '-0.04em',
				lineHeight: 0.85,
				usage: 'Figure, 5u: one number that carries the piece.'
			}),
			style('lede', u, 0.72, {
				fontWeight: 400,
				lineHeight: 1.35,
				usage: 'Lede, 0.72u, in muted ink.'
			}),
			style('body', u, 0.55, {
				fontWeight: 400,
				lineHeight: 1.4,
				usage: 'Body, 0.55u: cards, notes, footers.'
			}),
			style('label', u, 0.48, {
				family: 'mono',
				fontWeight: 500,
				letterSpacing: '0.12em',
				lineHeight: 1,
				usage: 'Label, 0.48u: Llama Mono, uppercase, muted.'
			}),
			style('mono', u, 0.5, {
				family: 'mono',
				fontWeight: 400,
				lineHeight: 1.2,
				usage: 'Mono, 0.5u: commands, URLs, repo ids, data.'
			})
		]
	});
	// type styles are named once per group; the page shows each group's sizes at its unit
	const groups = [
		scale(45, 'Square, portrait, story (u = 45px)'),
		scale(37.5, 'Landscape 16:9 (u = 37.5px)')
	].map((g, i) => ({
		...g,
		styles: g.styles.map((s) => ({ ...s, name: i ? `${s.name}-landscape` : s.name }))
	}));
	const FORMATS = [
		['square', 45, '1080 × 1080: Instagram, LinkedIn, X'],
		['portrait', 45, '1080 × 1350, 4:5: Instagram, LinkedIn'],
		[
			'story',
			45,
			'1080 × 1920, 9:16: stories and Shorts; keep clear of the top 220px and the bottom 320px'
		],
		['landscape', 37.5, '1600 × 900, 16:9: X, Bluesky, slides, video thumbnails'],
		['og', 26.25, '1200 × 630: link previews on llama.app'],
		['repo', 26.67, '1280 × 640: the GitHub social preview'],
		['poster', 51.67, '1240 × 1754: an A-series poster at 150 dpi'],
		['billboard', 50, '2400 × 800, 3:1: banners and screens at events']
	];

	return {
		color: { themes: [{ id: 'light', name: 'Light' }], tokens: color },
		meta: {
			note: 'Exported from tools/ui in llama.cpp (src/lib/styles/tokens/primitives.css, the communications grounds and type scale) by scripts/export-design-system.mjs.',
			package: 'tools/ui',
			repo: 'ggml-org/llama.cpp',
			source: 'github'
		},
		name: 'Llama',
		radius: {
			tokens: [
				{ name: 'radius-chip', usage: 'Chips and pills: fully round.', value: '999px' },
				{ name: 'radius-card', usage: 'Cards: 0.55u (25px at u = 45).', value: '25px' },
				{
					name: 'radius-inset',
					usage: 'Insets and terminals: 0.3u (13px at u = 45).',
					value: '13px'
				}
			]
		},
		spacing: {
			note: 'Every size is a multiple of the format’s unit u: the short side ÷ 24, or the long side ÷ 48 on formats wider than 2:1. The margin is 2u, the gutter ½u; 6 columns on square and tall formats, 12 on wide ones.',
			tokens: FORMATS.map(([name, u, usage]) => ({
				name: `unit-${name}`,
				usage: `${usage}. Margin ${2 * u}px.`,
				value: `${u}px`
			}))
		},
		type: {
			families: {
				mono: '"Llama Mono", ui-monospace, Menlo, monospace',
				sans: '"Llama Sans", system-ui, sans-serif'
			},
			fonts: [
				{ family: 'Llama Sans', file: 'fonts/LlamaSans.woff2', style: 'normal', weight: '400 700' },
				{ family: 'Llama Mono', file: 'fonts/LlamaMono.woff2', style: 'normal', weight: '400 700' }
			],
			groups
		},
		version: 1
	};
}

function readme(examples) {
	const brand = fs.readFileSync(path.join(ROOT, 'BRAND.md'), 'utf8');

	return `${brand.trim()}

## In this system

- **Logos** (assets): the icon, the Llama lockup and the llama.cpp lockup, each for light and dark grounds, with
  and without their own ground, and the app icon for light and dark grounds. Use them as they are.
- **Motifs** (assets): every motif as an SVG for light grounds (ink #111111) and dark ones (ink #fafafa), with its
  one cinder detail; quantised blocks and the dot grid as images.
- **Components**: one per key visual (${KEY_VISUALS.map((k) => k.name).join(', ')}), each with its guidelines and
  ${examples} example images at their real size, from the Storybook.
- **Fonts**: Llama Sans and Llama Mono. **Tokens**: the colours, the type scale at two units, the units per format
  and the radii.

The Storybook these come from: ${SPACE} (Communications, Brand, Philosophy). Exported from llama.cpp's
\`tools/ui\` by \`npm run export:design-system\`; change the Storybook or \`BRAND.md\`, not this system.
`;
}

function componentReadme(k, files) {
	return `# ${k.name}

${k.summary}

**Use it for:** ${k.use}

## Shape

${k.shape.map((s) => `- ${s}`).join('\n')}

## Rules

- Start from the format's unit: every size is a multiple of u (see the tokens' spacing notes), inside a margin of 2u
  and the platform's safe area.
- One accent: cinder on one thing only.
- Real copy, real numbers, the real product. Model makers by name, never their logos.

## In the Storybook

${k.stories.map((id) => `- ${STORY_PATHS[id]} (\`${id}\`)`).join('\n')}

## Examples in this folder

${files.map((f) => `- \`examples/${f}\``).join('\n')}
`;
}

function componentPreview(k, files) {
	const shown = files.filter((f) => !f.includes('billboard')).slice(0, 4);

	return `<!-- @dsCard group="Key visuals" height=420 subtitle="${k.summary.replace(/"/g, '&quot;')}" -->
<div class="examples">
${shown.map((f) => `	<img alt="${k.name}: ${f.replace(/\.(png|jpg)$/, '').replace(/-/g, ' ')}" src="../../components/${k.name}/examples/${f}" />`).join('\n')}
</div>
<style>
	.examples {
		display: flex;
		gap: 16px;
		align-items: flex-start;
		padding: 16px;
		overflow-x: auto;
	}
	.examples img {
		height: 360px;
		width: auto;
		border-radius: 8px;
		box-shadow: 0 0 0 1px rgb(0 0 0 / 0.1);
	}
</style>
<script></script>
`;
}

// the cover: the palette as blocks (ink, cinder, sand, the day sky, the night sky) and a dot grid at the unit,
// the name in Llama Sans
const COVER = `<!-- @dsCard height=300 -->
<div class="cover">
	<svg viewBox="0 0 480 300" class="art" aria-hidden="true">
		<!--
			blocks: night-sky-top (tall slab), day-sky-top, sand-50, neutral-950 (ink), art-cinder (the one accent, small)
			arrangement: one tall slab with satellites, bleeding off the top and right edges
			pattern: a dot grid at one unit (the grid every piece is built on, made visible); precise and technical
			steps: 24px pitch (a unit at a small scale), radius-card on the slab, a disc for cinder
		-->
		<rect class="night" height="230" rx="25" width="170" x="300" y="-30" />
		<rect class="day" height="120" rx="25" width="150" x="130" y="-30" />
		<rect class="sand" height="96" rx="25" width="150" x="130" y="110" />
		<rect class="ink" height="74" rx="25" width="230" x="240" y="220" />
		<circle class="cinder" cx="205" cy="258" r="22" />
		<g class="dots">${Array.from({ length: 6 * 5 }, (_, i) => `<circle cx="${318 + (i % 6) * 24}" cy="${18 + Math.floor(i / 6) * 24}" r="2.2" />`).join('')}</g>
	</svg>
	<div class="name">
		<h1>Llama</h1>
		<p>Your AI, on your computer: the brand and communications of Llama and llama.cpp.</p>
	</div>
</div>
<style>
	.cover {
		position: relative;
		width: 960px;
		height: 300px;
		background: var(--neutral-50);
	}
	.art {
		position: absolute;
		top: 0;
		right: 0;
		width: 480px;
		height: 300px;
	}
	.night { fill: var(--night-sky-top); }
	.day { fill: var(--day-sky-top); }
	.sand { fill: var(--sand-100); }
	.ink { fill: var(--neutral-950); }
	.cinder { fill: var(--art-cinder); }
	.dots circle { fill: var(--neutral-50); opacity: 0.55; }
	.name {
		position: absolute;
		left: 40px;
		bottom: 36px;
		max-width: 420px;
	}
	h1 {
		margin: 0;
		color: var(--neutral-950);
		font: 700 120px / 0.92 "Llama Sans", system-ui, sans-serif;
		letter-spacing: -0.035em;
	}
	p {
		margin: 12px 0 0;
		color: var(--neutral-600);
		font: 400 14px / 1.4 "Llama Sans", system-ui, sans-serif;
	}
</style>
<script></script>
`;
const MIME = {
	'.css': 'text/css',
	'.html': 'text/html',
	'.js': 'text/javascript',
	'.json': 'application/json',
	'.mjs': 'text/javascript',
	'.png': 'image/png',
	'.svg': 'image/svg+xml',
	'.woff2': 'font/woff2'
};

function serve(dir) {
	return new Promise((resolve) => {
		const server = http.createServer((req, res) => {
			const url = decodeURIComponent(new URL(req.url, 'http://x').pathname);

			let file = path.join(dir, url);

			if (!file.startsWith(dir)) return res.writeHead(403).end();

			if (fs.existsSync(file) && fs.statSync(file).isDirectory())
				file = path.join(file, 'index.html');

			if (!fs.existsSync(file)) return res.writeHead(404).end();

			res.writeHead(200, {
				'content-type': MIME[path.extname(file)] ?? 'application/octet-stream'
			});
			fs.createReadStream(file).pipe(res);
		});

		server.listen(0, '127.0.0.1', () => resolve(server));
	});
}

async function render(storybook) {
	const { chromium } = await import('playwright');
	const server = await serve(storybook);
	const base = `http://127.0.0.1:${server.address().port}`;
	const browser = await chromium.launch();
	const page = await browser.newPage({ viewport: { height: 1200, width: 1600 } });
	const open = async (id, args = '') => {
		await page.goto(`${base}/iframe.html?id=${id}&viewMode=story${args ? `&args=${args}` : ''}`);
		await page.waitForSelector('.post');
		await page.evaluate(() => document.fonts.ready);
		await page.waitForTimeout(400);
	};
	// motifs: each SVG with its computed colours, fonts and strokes written into it, for light and dark grounds;
	// the two drawn in CSS (quantised blocks, the dot grid) as images
	const motifs = [];

	for (const tone of ['light', 'dark']) {
		await open('communications-elements--motifs', `tone:${tone}`);

		const found = await page.evaluate(() => {
			const PROPS = [
				'fill',
				'fill-opacity',
				'stroke',
				'stroke-width',
				'stroke-opacity',
				'stroke-linecap',
				'stroke-linejoin',
				'stroke-dasharray',
				'opacity',
				'font-family',
				'font-size',
				'font-weight',
				'letter-spacing',
				'text-anchor',
				'dominant-baseline'
			];

			return [...document.querySelectorAll('figure')].map((figure) => {
				const label = figure.querySelector('figcaption span')?.textContent?.trim() ?? '';
				const svg = figure.querySelector('.post svg');

				if (!svg) return { label, svg: null };

				const clone = svg.cloneNode(true);
				const from = [svg, ...svg.querySelectorAll('*')];
				const to = [clone, ...clone.querySelectorAll('*')];

				from.forEach((el, i) => {
					const cs = getComputedStyle(el);
					const style = PROPS.map((p) => `${p}:${cs.getPropertyValue(p)}`).join(';');

					to[i].removeAttribute('class');
					to[i].setAttribute('style', style);
				});

				clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
				clone.setAttribute('width', '800');
				clone.setAttribute('height', '600');
				clone.removeAttribute('aria-hidden');

				return { label, svg: clone.outerHTML.replace(/<!--[\s\S]*?-->/g, '') };
			});
		});

		for (const [i, m] of found.entries()) {
			const name = `${slug(m.label)}-${tone}`;

			if (m.svg) {
				write(`assets/Motifs/${name}.svg`, `<?xml version="1.0" encoding="UTF-8"?>\n${m.svg}\n`);
			} else {
				const tile = (await page.$$('figure'))[i];

				await (
					await tile.$('.post')
				).screenshot({ path: path.join(PROJECT, `assets/Motifs/${name}.png`) });
			}

			motifs.push(name);
		}
	}

	// key visuals and the applications on them, each piece at its real size
	await page.addInitScript(() => {
		addEventListener('DOMContentLoaded', () => {
			const style = document.createElement('style');

			style.textContent = `.fit{height:auto!important;width:max-content!important;max-width:none!important;overflow:visible!important;box-shadow:none!important;border-radius:0!important}
				.post{transform:none!important}
				section{width:auto!important}
				section > div{display:flex!important;flex-direction:column!important;align-items:flex-start!important}`;
			document.head.append(style);
		});
	});

	// each piece is filed under the key visual its label names ("Roster, Square, light"), whichever story it is in
	const examples = Object.fromEntries(KEY_VISUALS.map((k) => [k.name, []]));
	const stories = [...new Set(KEY_VISUALS.flatMap((k) => k.stories))];

	for (const id of stories) {
		await open(id);

		const posts = await page.$$('.post');

		console.log(`${id}: ${posts.length} pieces`);

		for (const post of posts) {
			const label = await post.getAttribute('aria-label');
			const layout = label.split(',')[0].trim();

			if (!examples[layout]) continue;

			const prefix = id.includes('applications') ? `${slug(id.split('--')[1])}-` : '';
			const name = `${prefix}${slug(label)}`;
			const dir = path.join(PROJECT, `components/${layout}/examples`);

			let file = `${name}.png`;

			fs.mkdirSync(dir, { recursive: true });
			await post.screenshot({ path: path.join(dir, file) });

			// anything over the system's 512 KB cap for a plain file goes as a JPEG
			if (fs.statSync(path.join(dir, file)).size > 500_000) {
				fs.rmSync(path.join(dir, file));
				file = `${name}.jpg`;
				await post.screenshot({ path: path.join(dir, file), quality: 88, type: 'jpeg' });
			}

			examples[layout].push(file);
		}
	}

	await browser.close();
	server.close();

	return { examples, motifs };
}

function index(uploads, title) {
	const groups = ['Logos', 'Motifs'];
	const assetGroups = {};

	for (const group of groups) {
		const files = {};
		const order = [];

		for (const [rel, blob] of Object.entries(uploads).sort()) {
			if (!rel.startsWith(`assets/${group}/`)) continue;

			const name = rel.slice(`assets/${group}/`.length);
			const file = path.join(PROJECT, rel);
			const type = rel.endsWith('.svg') ? 'image/svg+xml' : 'image/png';

			files[
				name.replace(
					/[^A-Za-z0-9_./-]/g,
					(c) => `~${c.charCodeAt(0).toString(16).padStart(2, '0')}`
				)
			] = {
				blob: blob.replace(/^\/_blob\//, ''),
				name,
				size: fs.statSync(file).size,
				type
			};
			order.push(name);
		}

		assetGroups[group] = { files, name: group, order, tile: group === 'Logos' ? 'm' : 'l' };
	}

	return {
		assetGroups,
		blobs: {},
		createdOnFiles: { at: new Date().toISOString(), v: 1 },
		docs: { readme: 'project/README.md', sections: [] },
		groups,
		lastChange: {
			at: new Date().toISOString(),
			by: 'Claude',
			note: 'Exported from the Storybook and BRAND.md',
			via: 'Claude Code · ggml-org/llama.cpp tools/ui'
		},
		layout: 'files',
		libraries: [],
		namespace: 'Llama',
		sections: {},
		title,
		v: 3
	};
}

if (args[0] === 'index') {
	const uploads = JSON.parse(
		fs.readFileSync(path.resolve(flag('uploads', 'uploads.json')), 'utf8')
	);

	// the previews show their example images from the uploads: a preview cannot load the system's plain files
	for (const k of KEY_VISUALS) {
		const file = path.join(PROJECT, `components/${k.name}/preview.html`);
		const html = fs
			.readFileSync(file, 'utf8')
			.replace(/src="\.\.\/\.\.\/([^"]+)"/g, (m, rel) =>
				uploads[rel] ? `src="${uploads[rel]}"` : m
			);

		fs.writeFileSync(file, html);
	}

	write(
		'design-system.json',
		`${JSON.stringify(index(uploads, flag('title', 'Llama')), null, '\t')}\n`
	);
	console.log(`wrote ${path.relative(ROOT, path.join(PROJECT, 'design-system.json'))}`);
} else {
	const storybook = path.resolve(ROOT, flag('storybook', 'storybook-static'));

	if (!fs.existsSync(path.join(storybook, 'iframe.html'))) {
		console.log('building Storybook…');
		execSync(`npx storybook build -o '${storybook}' --quiet`, { cwd: ROOT, stdio: 'inherit' });
	}

	fs.rmSync(OUT, { force: true, recursive: true });
	write('tokens.json', `${JSON.stringify(tokens(), null, '\t')}\n`);

	for (const font of ['LlamaSans.woff2', 'LlamaMono.woff2'])
		write(`fonts/${font}`, fs.readFileSync(path.join(ROOT, 'src/lib/assets/fonts', font)));

	for (const [dir, file] of LOGO_FILES)
		write(`assets/Logos/${file}`, fs.readFileSync(path.join(ROOT, LOGOS, dir, file)));

	for (const [source, file] of APP_ICONS)
		write(
			`assets/Logos/${file}`,
			fs
				.readFileSync(path.join(ROOT, LOGOS, source), 'utf8')
				.replace(
					'width="1024" height="1024" viewBox="0 0 1024 1024"',
					'width="824" height="824" viewBox="100 100 824 824"'
				)
		);

	write(
		'assets/Logos/README.md',
		'# Logos\n\nThe icon and the two lockups, as they are: `-light` is the ink mark for light grounds, `-dark` the light mark for dark grounds, `-bg` with its own ground. Llama goes on any ground; llama.cpp on light and dark only. Never redraw, recolour or outline them.\n\nThe app icon (`app-icon-light.svg`, `app-icon-dark.svg`) is for Llama as a thing in a piece, such as a node in a diagram: the tile as the app ships it, its corners at 22.5% of its side and the icon at 67% of it, optically centred. Use it at about 2.4u with at least 1u clear round it; never draw a square round the icon instead.\n'
	);
	write(
		'assets/Motifs/README.md',
		'# Motifs\n\nLine drawings for pieces, in the ground’s ink with one cinder detail: `-light` for light grounds (ink #111111), `-dark` for dark ones (ink #fafafa). Models and maths: decision tree, neural network, tensor, gradient descent, vector space. Charts and fields: curve, quantised blocks, dot grid. Machines, isometric or flat: GPU, laptop, small and large workstation, server. A motif fills its area; it is never the accent itself beyond its own cinder detail.\n'
	);

	const { examples } = await render(storybook);
	const count = Object.values(examples).reduce((n, f) => n + f.length, 0);

	for (const k of KEY_VISUALS) {
		write(`components/${k.name}/README.md`, componentReadme(k, examples[k.name]));
		write(`components/${k.name}/preview.html`, componentPreview(k, examples[k.name]));
	}

	write('components/Cover/preview.html', COVER);
	// the previews need a bundle to run; the key visuals have no code of their own here, so it is empty
	write(
		'components/bundle.js',
		`/* @ds-bundle: {"format":4,"namespace":"Llama","components":[]} */\nwindow.Llama = {};\n`
	);
	// the images the previews show, to upload as assets before the index step
	write(
		'../preview-images.json',
		`${JSON.stringify(
			KEY_VISUALS.flatMap((k) =>
				[...componentPreview(k, examples[k.name]).matchAll(/src="\.\.\/\.\.\/([^"]+)"/g)].map(
					(m) => m[1]
				)
			),
			null,
			'\t'
		)}\n`
	);
	write('README.md', readme(count));

	console.log(
		`exported to ${path.relative(ROOT, OUT)}: ${count} examples, tokens, fonts, logos and motifs`
	);
}
