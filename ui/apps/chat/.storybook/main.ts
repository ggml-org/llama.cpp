import type { StorybookConfig } from '@storybook/sveltekit';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Llama UI's stories live with the library (ui/lib/ui/tests/stories), which runs and tests them in its own
// Storybook; this one shows them too, under Llama UI, next to Llama Chat's own. The library's front page stays its own.
const LLAMA_UI_STORIES = resolve(__dirname, '../../../lib/ui/tests/stories');

const config: StorybookConfig = {
	stories: [
		'../tests/stories/**/*.mdx',
		'../tests/stories/**/*.stories.@(js|ts|svelte)',
		{
			directory: '../../../lib/ui/tests/stories',
			files: '{foundations,components,app,models,mockups}/**/*.stories.svelte'
		}
	],
	addons: [
		'@storybook/addon-svelte-csf',
		'@chromatic-com/storybook',
		'@storybook/addon-vitest',
		'@storybook/addon-a11y',
		'@storybook/addon-docs',
		'@storybook/addon-mcp'
	],
	framework: '@storybook/sveltekit',
	viteFinal: async (config) => {
		config.server = config.server || {};
		config.server.fs = config.server.fs || {};
		config.server.fs.allow = [
			...(config.server.fs.allow || []),
			resolve(__dirname, '../tests'),
			LLAMA_UI_STORIES
		];
		// the library's stories import Storybook from where they live, where only this app's packages are installed
		config.resolve = config.resolve || {};
		config.resolve.dedupe = [
			...(config.resolve.dedupe || []),
			'@storybook/addon-svelte-csf',
			'storybook'
		];
		return config;
	}
};
export default config;
