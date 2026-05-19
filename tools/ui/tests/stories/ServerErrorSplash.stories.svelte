<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { expect } from 'storybook/test';
	import ServerErrorSplash from '$lib/components/app/server/ServerErrorSplash.svelte';

	const { Story } = defineMeta({
		title: 'Components/Server/ServerErrorSplash',
		component: ServerErrorSplash,
		parameters: {
			layout: 'fullscreen'
		}
	});
</script>

<Story
	name="ApiKeyInput"
	args={{
		error: '401 Unauthorized',
		showRetry: false
	}}
	play={async ({ canvas, userEvent }) => {
		const showInputButton = await canvas.findByRole('button', { name: 'Enter API Key' });
		await userEvent.click(showInputButton);

		const apiKeyInput = await canvas.findByLabelText('API Key');
		await expect(apiKeyInput).toHaveAttribute('type', 'password');
		await expect(apiKeyInput).toHaveAttribute('autocomplete', 'off');

		const showValueButton = await canvas.findByRole('button', { name: 'Show API key' });
		await userEvent.click(showValueButton);

		await expect(apiKeyInput).toHaveAttribute('type', 'text');
	}}
/>
