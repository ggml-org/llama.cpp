import { mdsvex } from 'mdsvex';
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

// CMake sets LLAMA_UI_OUT_DIR to the staging dir under the build tree; manual
// `npm run build` runs without the env var default to ./dist.
const outDir = process.env.LLAMA_UI_OUT_DIR ?? './dist';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	preprocess: [vitePreprocess(), mdsvex()],

	kit: {
		paths: {
			relative: true
		},
		router: { type: 'hash' },
		adapter: adapter({
			pages: outDir,
			assets: outDir,
			fallback: 'index.html',
			precompress: false,
			strict: true
		}),
		output: {
			bundleStrategy: 'single'
		},
		alias: {
			$styles: 'src/styles'
		}
	},

	extensions: ['.svelte', '.svx']
};

export default config;
