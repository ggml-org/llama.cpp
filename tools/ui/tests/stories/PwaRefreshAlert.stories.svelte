<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import PwaRefreshAlert from '$lib/components/pwa/PwaRefreshAlert.svelte';
	import { expect } from 'storybook/test';

	const { Story } = defineMeta({
		component: PwaRefreshAlert,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/PwaRefreshAlert'
	});
</script>

<Story
	args={{ needRefresh: true, updateServiceWorker: () => console.log('reload') }}
	name="Default"
	play={async ({ canvas }) => {
		const title = canvas.getByText('Update available');

		await expect(title).toBeInTheDocument();

		const description = canvas.getByText(/A new version is available/);

		await expect(description).toBeInTheDocument();

		const button = canvas.getByRole('button', { name: 'Reload' });

		await expect(button).toBeInTheDocument();
	}}
/>
