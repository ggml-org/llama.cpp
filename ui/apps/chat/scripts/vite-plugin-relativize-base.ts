import { BUILD_CONFIG } from '../src/lib/constants/pwa.constants.ts';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'path';
import type { Plugin } from 'vite';

const OUTPUT_DIR = process.env.LLAMA_UI_OUT_DIR ?? BUILD_CONFIG.OUTPUT_DIR;

function rewrite(path: string, pairs: [string, string][]): void {
	if (!existsSync(path)) {
		return;
	}

	const text = readFileSync(path, 'utf-8');

	let out = text;

	for (const [from, to] of pairs) {
		out = out.split(from).join(to);
	}

	if (out !== text) {
		writeFileSync(path, out, 'utf-8');
	}
}

/**
 * Relativize SvelteKit absolute base refs so the build is relocatable under any subpath.
 *
 * SvelteKit bakes root absolute /_app/ paths into the SPA fallback because paths.relative
 * does not apply to a depth agnostic fallback page. Rewriting to ./_app/ lets a plain
 * recursive copy of the output into /any/subdir/ resolve assets against the document URL.
 * Runs after adapter-static writes index.html.
 */
export function relativizeBasePlugin(): Plugin {
	return {
		apply: 'build',
		buildApp: {
			// post order runs after the SvelteKit adapter has written the output
			async handler() {
				try {
					const outDir = resolve(OUTPUT_DIR);

					// index.html: modulepreload, stylesheet and bootstrap import reference "/_app/
					rewrite(resolve(outDir, 'index.html'), [['"/_app/', '"./_app/']]);
					console.log('Relativized base refs in index.html');
				} catch (error) {
					console.error('Failed to relativize base refs:', error);
				}
			},
			order: 'post'
		},
		name: 'llamacpp:relativize-base'
	};
}
