<script lang="ts" module>
	import { fakeAccountWith } from '../fixtures/hf-fake-account.svelte.ts';
	import HfShareDialogStoryHost from './HfShareDialogStoryHost.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { expect, screen, userEvent, waitFor } from 'storybook/test';

	// the dialog that shares a conversation to a Hugging Face dataset; an account with no calls out of it
	const { Story } = defineMeta({
		component: HfShareDialogStoryHost,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/Dialogs/HfShareDialog'
	});
</script>

<Story
	args={{ account: fakeAccountWith('agent-traces', 'llama-chat-traces') }}
	name="DatasetsToAddTo"
/>

<Story args={{ account: fakeAccountWith() }} name="NoDatasetYet" />

<Story
	args={{ account: fakeAccountWith('a', 'b', 'c', 'd', 'e') }}
	name="MoreDatasets"
	play={async () => {
		await userEvent.click(await screen.findByRole('button', { name: 'Show 2 more' }));

		const tile = await screen.findByRole('button', { name: /^e\b/ });

		// the dialog fades in: it is visible once it has finished
		await waitFor(() => expect(tile).toBeVisible());
	}}
/>

<Story
	args={{ account: fakeAccountWith('agent-traces') }}
	name="Shared"
	play={async () => {
		await userEvent.click(await screen.findByRole('button', { name: 'Share' }));

		const title = await screen.findByText('Shared to Hugging Face');

		await waitFor(() => expect(title).toBeVisible());
	}}
/>
