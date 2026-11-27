import type { Preview } from '@storybook/sveltekit';
import '../src/app.css';
import './docs.css';
import { applyColorScheme } from './color-scheme';
import ModeWatcherDecorator from './decorators/ModeWatcherDecorator.svelte';
import TooltipProviderDecorator from './decorators/TooltipProviderDecorator.svelte';

applyColorScheme();
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyColorScheme);

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

		options: {
			// the package's own sections, in the order Llama Chat's Storybook (ui/apps/chat) shows them under Llama UI:
			// the foundations, the base components, the app components, then the Model UI and the mockups
			storySort: {
				order: [
					'Introduction',
					'Llama UI',
					[
						'Foundations',
						['Colors', 'Typography', 'Sizing', 'Shapes', 'Elevation', 'Surfaces'],
						'Components',
						['Actions', 'Forms', 'Display', 'Overlays', 'Navigation'],
						'App components',
						'Models',
						'Mockups'
					]
				]
			}
		},

		a11y: {
			// 'todo' shows violations in the Accessibility panel and the test UI without failing a test; 'error' would
			// fail CI on them. Not yet 'error': about half the stories fail it, most on colour contrast (the muted
			// foreground on the light surfaces, below 4.5:1) and nested interactive controls, which need design changes
			// rather than story fixes. Switch once those are settled
			test: 'todo'
		}
	},
	decorators: [
		// `parameters.colorScheme` fixes a story to light or dark, whatever the viewer chose (a scene in a store screenshot)
		(story, context) => ({
			Component: ModeWatcherDecorator,
			props: {
				children: story,
				forced: context.parameters.colorScheme
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
