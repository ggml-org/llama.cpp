<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatScreenForm from '$lib/components/app/chat/ChatScreen/ChatScreenForm.svelte';
	import { ATTACHMENT_TOOLTIP_TEXT } from '$lib/constants';
	import { ServerRole } from '$lib/enums';
	import { serverStore } from '$lib/stores';
	import type { ApiLlamaCppServerProps } from '$lib/types';
	import { expect, screen, waitFor } from 'storybook/test';

	/**
	 * The add menu mounts the reasoning submenu only outside router mode, and the
	 * dev server proxies /props to whichever server happens to be running, so pin
	 * the mode this story asserts instead of inheriting it from the environment.
	 */
	function pinSingleModelMode(): void {
		serverStore.props = {
			...(serverStore.props ?? {}),
			role: ServerRole.MODEL
		} as ApiLlamaCppServerProps;

		serverStore.role = ServerRole.MODEL;
	}

	const { Story } = defineMeta({
		component: ChatScreenForm,
		parameters: {
			layout: 'centered'
		},
		tags: ['!dev'],
		title: 'Components/ChatScreen/ChatScreenForm/Accessibility'
	});
</script>

<Story
	args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
	name="AddButtonSingleTabStop"
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
	args={{ class: 'max-w-[56rem] w-[calc(100vw-2rem)]' }}
	name="AddDropdownFocusesFirstEnabled"
	play={async ({ canvas, userEvent }) => {
		pinSingleModelMode();

		const trigger = await canvas.findByRole('button', { name: ATTACHMENT_TOOLTIP_TEXT });

		trigger.focus();
		await userEvent.keyboard('{Enter}');
		await screen.findByRole('menu');

		await waitFor(() => {
			expect(document.activeElement).toHaveTextContent('Reasoning');
		});
	}}
/>
