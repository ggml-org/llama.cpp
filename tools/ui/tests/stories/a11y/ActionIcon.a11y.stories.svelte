<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Copy } from '@lucide/svelte';
	import ActionIcon from '$lib/components/app/actions/ActionIcon.svelte';
	import { expect } from 'storybook/test';

	const { Story } = defineMeta({
		title: 'Components/ActionIcon/Accessibility',
		component: ActionIcon,
		parameters: {
			layout: 'centered'
		},
		tags: ['!dev']
	});
</script>

<Story
	name="SingleTabStop"
	args={{ icon: Copy, tooltip: 'Copy', onclick: () => {} }}
	play={async ({ canvas, userEvent }) => {
		const button = await canvas.findByRole('button', { name: 'Copy' });

		button.focus();
		await expect(button).toHaveFocus();

		await userEvent.tab();

		await expect(button).not.toHaveFocus();
	}}
/>
