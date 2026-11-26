<script lang="ts" module>
	import ChatFormAddSheetStoryHost from './ChatFormAddSheetStoryHost.svelte';
	import { resetPointerEvents } from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ATTACHMENT_MENU_TEXT } from '#lib/constants/index.ts';
	import { screen } from 'storybook/test';

	// a phone's + menu: a short sheet of grouped rows. Files and the camera open the phone's own
	// pickers, the system message and working directory close the sheet, and Tools pushes pages:
	// the tool groups, then one group's tools, each with a switch
	const { Story } = defineMeta({
		beforeEach: resetPointerEvents,
		component: ChatFormAddSheetStoryHost,
		globals: { viewport: { isRotated: false, value: 'mobile2' } },
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Llama Chat/Chat/ChatForm/ChatFormActionAddDrawer'
	});

	const WORKING_DIRECTORY = '~/projects/llama.cpp';

	async function openSheet({
		canvas,
		userEvent
	}: {
		canvas: { getByRole: (role: string, options: { name: string }) => Element };
		userEvent: { click: (element: Element) => Promise<void> };
	}) {
		await userEvent.click(canvas.getByRole('button', { name: ATTACHMENT_MENU_TEXT.ADD_TO_CHAT }));
		// the row's name gains the kinds of file the model takes
		await screen.findByRole('button', { name: new RegExp(ATTACHMENT_MENU_TEXT.ADD_FILES) });
	}
</script>

<Story name="Menu" play={openSheet} />

<Story args={{ hasVisionModality: true }} name="With the camera" play={openSheet} />

<Story
	args={{ workingDirectory: WORKING_DIRECTORY }}
	name="With a working directory"
	play={openSheet}
/>
