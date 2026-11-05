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
			// sidebar sections: the introduction, the brand (logo, type, colour), then the UI: its foundations
			// first, the full chat screen leading its folder
			storySort: {
				order: [
					'Introduction',
					['Design guidelines'],
					'Brand',
					[
						'Assets',
						['Logo', 'Social profiles'],
						'Logo',
						['Overview', 'Icon', 'Lockup proposal'],
						'Typography',
						[
							'Overview',
							'Tester',
							'Llama Sans',
							['User Interface', 'Running text', 'Display', 'Glyphs'],
							'Llama Mono'
						],
						'Colors',
						['Palette', 'Communications', 'Palette proposal']
					],
					'UI',
					[
						'Foundations',
						['Colors', 'Typography', 'Sizing', 'Shapes', 'Elevation', 'Surfaces'],
						'Actions',
						'Forms',
						'Display',
						'Overlays',
						'Navigation',
						'Patterns',
						'Blocks',
						[
							'Chat',
							['ChatScreen', ['ChatScreen', '*']],
							'Content',
							'Models',
							'MCP',
							'Navigation',
							'Settings',
							['DialogSettingsChat', 'Navigation', 'Sections'],
							'App'
						],
						'Proposals'
					]
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
	// read by tests/stories/pace.ts; the test runner always plays at full speed
	globalTypes: {
		interactionPace: {
			description: 'Speed of the interactions a story plays',
			toolbar: {
				dynamicTitle: true,
				icon: 'timer',
				items: [
					{ title: 'Human pace', value: 'human' },
					{ title: 'Instant', value: 'instant' }
				],
				title: 'Interaction pace'
			}
		},
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
		interactionPace: 'human',
		surface: 'paper'
	},
	decorators: [
		(story, context) => {
			document.documentElement.classList.toggle('theme-matte', context.globals.surface === 'matte');

			return story();
		},
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
