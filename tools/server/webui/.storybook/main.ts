import type { StorybookConfig } from '@storybook/sveltekit';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const config: StorybookConfig = {
	stories: ['../tests/stories/**/*.mdx', '../tests/stories/**/*.stories.@(js|ts|svelte)'],
	addons: [
		'@storybook/addon-svelte-csf',
		'@chromatic-com/storybook',
		'@storybook/addon-vitest',
		'@storybook/addon-a11y',
		'@storybook/addon-docs'
	],
	framework: '@storybook/sveltekit',
	viteFinal: async (config) => {
		config.server = config.server || {};
		config.server.fs = config.server.fs || {};
		config.server.fs.allow = [...(config.server.fs.allow || []), resolve(__dirname, '../tests')];
		return config;
	}
};
export default config;
