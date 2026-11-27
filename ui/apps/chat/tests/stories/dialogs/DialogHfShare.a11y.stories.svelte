<script lang="ts" module>
	import { fakeAccountWith } from '../fixtures/hf-fake-account.svelte.ts';
	import HfShareDialogStoryHost from './HfShareDialogStoryHost.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { expect, screen, waitFor, within } from 'storybook/test';

	const { Story } = defineMeta({
		component: HfShareDialogStoryHost,
		parameters: {
			layout: 'centered'
		},
		tags: ['!dev'],
		title: 'Llama Chat/Dialogs/HfShareDialog/Accessibility'
	});
</script>

<Story
	args={{ account: fakeAccountWith('alpha') }}
	name="DialogIsNamedAndTakesFocus"
	play={async () => {
		const dialog = await screen.findByRole('alertdialog', { name: /Share to Hugging Face/ });

		await waitFor(() => expect(dialog).toContainElement(document.activeElement as HTMLElement));
	}}
/>

<Story
	args={{ account: fakeAccountWith('alpha', 'beta') }}
	name="DatasetsAreOneGroupOfToggleButtons"
	play={async () => {
		const group = await screen.findByRole('group', { name: 'Dataset' });
		const alpha = await within(group).findByRole('button', { name: /^alpha\b/ });
		const beta = await within(group).findByRole('button', { name: /^beta\b/ });

		await expect(alpha).toHaveAttribute('aria-pressed', 'true');
		await expect(beta).toHaveAttribute('aria-pressed', 'false');
	}}
/>

<Story
	args={{ account: fakeAccountWith('alpha', 'beta') }}
	name="KeyboardPicksATile"
	play={async ({ userEvent }) => {
		const dialog = await screen.findByRole('alertdialog');
		const alpha = await screen.findByRole('button', { name: /^alpha\b/ });
		const beta = await screen.findByRole('button', { name: /^beta\b/ });

		// the dialog moves focus itself as it opens: start from the tile once that has settled
		await waitFor(() => expect(dialog).toContainElement(document.activeElement as HTMLElement));
		alpha.focus();
		await expect(alpha).toHaveFocus();
		await userEvent.tab();
		await expect(beta).toHaveFocus();

		await userEvent.keyboard('{Enter}');
		await waitFor(() => expect(beta).toHaveAttribute('aria-pressed', 'true'));
		await expect(alpha).toHaveAttribute('aria-pressed', 'false');
	}}
/>

<Story
	args={{ account: fakeAccountWith() }}
	name="NewDatasetNameIsLabelled"
	play={async () => {
		const field = await screen.findByLabelText('New dataset name');

		// the dialog fades in: it is visible once it has finished
		await waitFor(() => expect(field).toBeVisible());
	}}
/>
