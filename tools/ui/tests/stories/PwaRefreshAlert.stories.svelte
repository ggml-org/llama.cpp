<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import PwaRefreshAlert from '$lib/components/pwa/PwaRefreshAlert.svelte';
	import { expect } from 'storybook/test';

	const { Story } = defineMeta({
		title: 'Components/PwaRefreshAlert',
		component: PwaRefreshAlert,
		parameters: {
			layout: 'centered'
		}
	});
</script>

<Story
	name="Default"
	args={{ needRefresh: true, updateServiceWorker: () => console.log('reload') }}
	play={async ({ canvas }) => {
		const title = canvas.getByText('Update available');
		await expect(title).toBeInTheDocument();

		const description = canvas.getByText(/A new version is available/);
		await expect(description).toBeInTheDocument();

		const button = canvas.getByRole('button', { name: 'Reload' });
		await expect(button).toBeInTheDocument();
	}}
/>

<Story
	name="Hidden"
	args={{ needRefresh: false, updateServiceWorker: () => console.log('reload') }}
	play={async ({ canvas }) => {
		const title = canvas.queryByText('Update available');
		await expect(title).not.toBeInTheDocument();
	}}
/>

<Story
	name="ClickReload"
	args={{
		needRefresh: true,
		updateServiceWorker: () => console.log('reload')
	}}
	play={async ({ canvas, userEvent }) => {
		const button = canvas.getByRole('button', { name: 'Reload' });
		await expect(button).toBeInTheDocument();

		await userEvent.click(button);

		const title = canvas.queryByText('Update available');
		await expect(title).not.toBeInTheDocument();

		const reloadBtn = canvas.queryByRole('button', { name: 'Reload' });
		await expect(reloadBtn).not.toBeInTheDocument();
	}}
/>
