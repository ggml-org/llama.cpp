import type { Preview } from '@storybook/sveltekit';
import '../src/app.css';
import ModeWatcherDecorator from './decorators/ModeWatcherDecorator.svelte';
import TooltipProviderDecorator from './decorators/TooltipProviderDecorator.svelte';

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
			// sidebar sections; foundations lead the design system, the full chat screen leads its folder
			storySort: {
				order: [
					'Introduction',
					['Overview', 'Design guidelines'],
					'Brand',
					['Logo', 'App icons'],
					'Design system',
					[
						'Foundations',
						['Colors', 'Typography', 'Sizing', 'Shapes', 'Elevation', 'Surfaces'],
						'Actions',
						'Forms',
						'Display',
						'Overlays',
						'Navigation',
						'Patterns',
						'Proposals'
					],
					'Chat',
					['ChatScreen', ['ChatScreen', '*']],
					'Content',
					'Models',
					'MCP',
					'Navigation',
					'Settings',
					'App'
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
		}
	},
	initialGlobals: {
		interactionPace: 'human'
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
