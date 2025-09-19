import { mdsvex } from 'mdsvex';
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	preprocess: [vitePreprocess(), mdsvex()],
	kit: {
		paths: {
			// Base path for deployment in subdirectories (e.g., '/llama' for hosting at example.com/llama)
			// Leave empty ('') for root deployment (default behavior)
			base: process.env.BASE_PATH || ''
		},
		adapter: adapter({
			pages: '../public',
			assets: '../public',
			fallback: 'index.html',
			precompress: false,
			strict: true
		}),
		output: {
			bundleStrategy: 'inline'
		}
	},
	extensions: ['.svelte', '.svx']
};

export default config;
