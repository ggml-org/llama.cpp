import type { Preview } from '@storybook/sveltekit';
import '../src/app.css';
import ModeWatcherDecorator from './ModeWatcherDecorator.svelte';
import TooltipProviderDecorator from './TooltipProviderDecorator.svelte';

const preview: Preview = {
	parameters: {
		controls: {
			matchers: {
				color: /(background|color)$/i,
				date: /Date$/i
			}
		},
		backgrounds: {
			disable: true
		}
	},
	decorators: [
		(story) => ({
			Component: ModeWatcherDecorator,
			props: {
				children: story
			}
		}),
		(story) => ({
			Component: TooltipProviderDecorator,
			props: {
				children: story
			}
		})
	]
};

export default preview;
