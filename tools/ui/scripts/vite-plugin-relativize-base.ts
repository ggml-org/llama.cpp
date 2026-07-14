import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { resolve } from 'path';
import type { Plugin } from 'vite';
import { BUILD_CONFIG } from '../src/lib/constants/pwa';

let processed = false;

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
 * Runs after adapter-static writes index.html and the PWA plugin writes sw.js, deferred the
 * same way as buildInfoPlugin so the emitted files exist.
 */
export function relativizeBasePlugin(): Plugin {
	return {
		name: 'llamacpp:relativize-base',
		apply: 'build',
		closeBundle() {
			setTimeout(() => {
				try {
					if (processed) return;
					processed = true;

					const outDir = resolve(OUTPUT_DIR);

					// index.html: modulepreload, stylesheet and bootstrap import reference "/_app/
					rewrite(resolve(outDir, 'index.html'), [['"/_app/', '"./_app/']]);

					// sw.js: the only absolute entries are the navigate fallback precache key and handler
					rewrite(resolve(outDir, 'sw.js'), [
						['{url:"/"', '{url:"./"'],
						['createHandlerBoundToURL("/"', 'createHandlerBoundToURL("./"']
					]);

					console.log('Relativized base refs in index.html and sw.js');
				} catch (error) {
					console.error('Failed to relativize base refs:', error);
				}
			}, 100);
		}
	};
}
