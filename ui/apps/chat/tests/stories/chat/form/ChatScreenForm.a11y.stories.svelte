<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatScreenForm from '#lib/components/app/chat/ChatScreen/ChatScreenForm.svelte';
	import { attachmentTooltipText } from '#lib/constants/index.ts';
	import { expect, screen, waitFor } from 'storybook/test';

	const { Story } = defineMeta({
		component: ChatScreenForm,
		parameters: {
			layout: 'centered'
		},
		tags: ['!dev'],
		title: 'Llama Chat/Chat/ChatForm/ChatScreenForm/Accessibility'
	});
</script>

<Story
	args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
	name="AddButtonSingleTabStop"
	play={async ({ canvas, userEvent }) => {
		const editor = await canvas.findByRole('textbox');

		await userEvent.clear(editor);
		await userEvent.type(editor, 'What is the meaning of life?');

		const trigger = await canvas.findByRole('button', { name: attachmentTooltipText(true) });

		trigger.focus();
		await expect(trigger).toHaveFocus();

		await userEvent.tab();

		await expect(trigger).not.toHaveFocus();
	}}
/>

<Story
	args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
	name="AddDropdownFocusesFirstEnabled"
	play={async ({ canvas, userEvent }) => {
		const trigger = await canvas.findByRole('button', { name: attachmentTooltipText(true) });

		trigger.focus();
		await userEvent.keyboard('{Enter}');
		await screen.findByRole('menu');

		await waitFor(() => {
			expect(document.activeElement).toHaveTextContent('Add files');
		});
	}}
/>
