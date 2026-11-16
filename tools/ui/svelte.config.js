import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import { mdsvex } from 'mdsvex';

// CMake sets LLAMA_UI_OUT_DIR to the staging dir under the build tree; manual
// `npm run build` runs without the env var default to ./dist.
const outDir = process.env.LLAMA_UI_OUT_DIR ?? './dist';
/** @type {import('@sveltejs/kit').Config} */
const config = {
	extensions: ['.svelte', '.svx'],

	kit: {
		adapter: adapter({
			assets: outDir,
			fallback: 'index.html',
			pages: outDir,
			precompress: false,
			strict: true
		}),
		output: {
			bundleStrategy: 'single'
		},
		paths: {
			relative: true
		},
		router: { type: 'hash' }
	},

	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	// mdsvex is for content pages we write ourselves (e.g. release notes reachable from the
	// chat app): .svx files compiled to components at build time. No page uses it yet.
	// It can't render chat input or messages, which are typed at runtime and go through
	// the remark/rehype pipeline in MarkdownContent instead.
	preprocess: [vitePreprocess(), mdsvex()]
};

export default config;
