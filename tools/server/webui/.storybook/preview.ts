import type { Preview } from '@storybook/sveltekit';
import '../src/app.css';
import ModeWatcherDecorator from './ModeWatcherDecorator.svelte';

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
		})
	]
};

export default preview;
