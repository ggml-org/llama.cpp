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
			// the same sections as Llama UI's Storybook in llama.cpp (ui/apps/chat), with only what this package has so far:
			// the introduction, the UI with its foundations first and its blocks last, then the Scrapbook
			storySort: {
				order: [
					'Introduction',
					'UI',
					[
						'Foundations',
						['Colors', 'Typography', 'Sizing', 'Shapes', 'Elevation', 'Surfaces'],
						'Actions',
						'Forms',
						'Display',
						'Overlays',
						'Blocks',
						['Models']
					],
					'Scrapbook',
					['About', 'UI', 'Communications', 'Brand']
				]
			}
		},

		a11y: {
			// 'todo' - show a11y violations in the test UI only
			// 'error' - fail CI on a11y violations
			// 'off' - skip a11y checks entirely
			test: 'todo'
		}
	},
	globalTypes: {
		// the light surface theme: Paper (white panels on a grey page) or Matte (grey glass on white)
		surface: {
			description: 'Light surface theme',
			toolbar: {
				dynamicTitle: true,
				icon: 'paintbrush',
				items: [
					{ title: 'Paper', value: 'paper' },
					{ title: 'Matte', value: 'matte' }
				],
				title: 'Surface'
			}
		}
	},
	initialGlobals: {
		surface: 'paper'
	},
	decorators: [
		(story, context) => {
			document.documentElement.classList.toggle('theme-matte', context.globals.surface === 'matte');

			return story();
		},
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
