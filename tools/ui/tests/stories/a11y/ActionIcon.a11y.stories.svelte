<script module lang="ts">
	import { Copy } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ActionIcon from '$lib/components/app/actions/ActionIcon.svelte';
	import { expect } from 'storybook/test';

	const { Story } = defineMeta({
		component: ActionIcon,
		parameters: {
			layout: 'centered'
		},
		tags: ['!dev'],
		title: 'Components/ActionIcon/Accessibility'
	});
</script>

<Story
	asChild
	name="SingleTabStop"
	play={async ({ canvas, userEvent }) => {
		const before = await canvas.findByRole('button', { name: 'before' });
		const target = await canvas.findByRole('button', { name: 'Copy' });

		before.focus();
		await userEvent.tab();

		await expect(target).toHaveFocus();
	}}
>
	<div>
		<button type="button">before</button>
		<ActionIcon icon={Copy} tooltip="Copy" onclick={() => {}} />
	</div>
</Story>
