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
		'@storybook/addon-docs',
		'@storybook/addon-mcp'
	],
	framework: '@storybook/sveltekit',
	viteFinal: async (config) => {
		config.server = config.server || {};
		config.server.fs = config.server.fs || {};
		config.server.fs.allow = [...(config.server.fs.allow || []), resolve(__dirname, '../tests')];
		// the brand files the stories offer to download stay files: Vite inlines one under 4 KB as a data: URL,
		// which a download link does not reliably save, and which can navigate the story's frame away instead
		config.build = config.build || {};
		config.build.assetsInlineLimit = (file) =>
			file.includes('/tests/stories/brand/assets/files/') ? false : undefined;
		return config;
	}
};
export default config;
