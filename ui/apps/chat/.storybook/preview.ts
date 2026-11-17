import '../src/app.css';
import { MEDIA_QUERIES } from '#lib/constants/pwa.constants.ts';
import { SurfaceTheme } from '#lib/enums/ui.enums.ts';
import { applySurfaceTheme } from '#lib/utils/css.ts';
import './docs.css';
import { applyColorScheme } from './color-scheme';
import ModeWatcherDecorator from './decorators/ModeWatcherDecorator.svelte';
import TooltipProviderDecorator from './decorators/TooltipProviderDecorator.svelte';
import type { Preview } from '@storybook/sveltekit';

applyColorScheme();
window.matchMedia(MEDIA_QUERIES.PREFERS_DARK).addEventListener('change', applyColorScheme);

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
			// sidebar sections: the introduction and the guidelines for agents, then the two layers as the code has them.
			// Llama UI is the library (ui/lib/ui): its foundations, base components, app components, Model UI and mockups.
			// Llama Chat is the app (ui/apps/chat): its own foundations, then one folder per src/lib/components/app/<area>,
			// the full chat screen leading Chat. The brand and communications are in llama-brand's Storybook
			storySort: {
				order: [
					'Introduction',
					'Agents',
					['Design guidelines'],
					'Llama UI',
					[
						'Foundations',
						['Colors', 'Typography', 'Sizing', 'Shapes', 'Elevation', 'Surfaces'],
						'Components',
						['Actions', 'Forms', 'Display', 'Overlays', 'Navigation'],
						'App components',
						'Models',
						'Mockups'
					],
					'Llama Chat',
					[
						'Foundations',
						['Safe areas', 'Brand'],
						'Chat',
						[
							'ChatScreen',
							['ChatScreen', '*'],
							'ChatForm',
							'ChatMessages',
							'ChatAttachments',
							'ChatTabs'
						],
						'Content',
						'Models',
						'MCP',
						'Navigation',
						'Settings',
						['DialogSettingsChat', 'Navigation', 'Sections'],
						'Dialogs',
						'Forms',
						'Actions',
						'Badges',
						'Misc',
						'Servers',
						'PWA'
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
					{ title: 'Paper', value: SurfaceTheme.PAPER },
					{ title: 'Matte', value: SurfaceTheme.MATTE }
				],
				title: 'Surface'
			}
		}
	},
	initialGlobals: {
		interactionPace: 'human',
		surface: SurfaceTheme.PAPER
	},
	decorators: [
		(story, context) => {
			applySurfaceTheme(context.globals.surface);

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
