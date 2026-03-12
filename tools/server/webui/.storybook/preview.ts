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
			disabled: true
		},

		a11y: {
			// 'todo' - show a11y violations in the test UI only
			// 'error' - fail CI on a11y violations
			// 'off' - skip a11y checks entirely
			test: 'todo'
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
