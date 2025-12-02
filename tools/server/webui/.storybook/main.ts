import type { StorybookConfig } from '@storybook/sveltekit';

const config: StorybookConfig = {
	stories: ['../tests/stories/**/*.mdx', '../tests/stories/**/*.stories.@(js|ts|svelte)'],
	addons: [
		'@storybook/addon-svelte-csf',
		'@chromatic-com/storybook',
		'@storybook/addon-docs',
		'@storybook/addon-a11y',
		'@storybook/addon-vitest'
	],
	framework: {
		name: '@storybook/sveltekit',
		options: {}
	}
};
export default config;
