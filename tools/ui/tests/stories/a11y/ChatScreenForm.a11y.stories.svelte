<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatScreenForm from '$lib/components/app/chat/ChatScreen/ChatScreenForm.svelte';
	import { expect, screen, waitFor } from 'storybook/test';
	import { ATTACHMENT_TOOLTIP_TEXT } from '$lib/constants';

	const { Story } = defineMeta({
		title: 'Components/ChatScreen/ChatScreenForm/Accessibility',
		component: ChatScreenForm,
		parameters: {
			layout: 'centered'
		},
		tags: ['!dev']
	});
</script>

<Story
	name="AddButtonSingleTabStop"
	args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
	play={async ({ canvas, userEvent }) => {
		const textarea = await canvas.findByRole('textbox');
		await userEvent.clear(textarea);
		await userEvent.type(textarea, 'What is the meaning of life?');

		const trigger = await canvas.findByRole('button', { name: ATTACHMENT_TOOLTIP_TEXT });

		trigger.focus();
		await expect(trigger).toHaveFocus();

		await userEvent.tab();

		await expect(trigger).not.toHaveFocus();
	}}
/>

<Story
	name="AddDropdownFocusesFirstEnabled"
	args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
	play={async ({ canvas, userEvent }) => {
		const trigger = await canvas.findByRole('button', { name: ATTACHMENT_TOOLTIP_TEXT });

		trigger.focus();
		await userEvent.keyboard('{Enter}');
		await screen.findByRole('menu');

		await waitFor(() => {
			expect(document.activeElement).toHaveTextContent('Text Files');
		});
	}}
/>
