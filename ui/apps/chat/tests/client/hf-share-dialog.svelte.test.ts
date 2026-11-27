// The share dialog of Llama Hugging Face, against an account it does not call out of: which
// datasets it offers as tiles, and that its pick never points at one the account cannot add to.

import HfShareDialogStoryHost from '../stories/dialogs/HfShareDialogStoryHost.svelte';
import {
	FAKE_OWNER,
	type FakeAccount,
	fakeAccountWith
} from '../stories/fixtures/hf-fake-account.svelte.ts';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

function renderDialog(account: FakeAccount) {
	render(HfShareDialogStoryHost, { account });
}

const tile = (name: string) => page.getByRole('button', { name: new RegExp(`^${name}\\b`) });

describe('HfShareDialog datasets', () => {
	afterEach(() => vi.restoreAllMocks());

	it('offers the datasets as tiles, with the first picked, and a tile for a new one', async () => {
		renderDialog(fakeAccountWith('alpha', 'beta'));

		await expect.element(tile('alpha')).toHaveAttribute('aria-pressed', 'true');
		await expect.element(tile('beta')).toHaveAttribute('aria-pressed', 'false');
		await expect.element(page.getByRole('button', { name: /^New dataset/ })).toBeVisible();
	});

	it('goes straight to naming a new dataset when there is none to add to', async () => {
		renderDialog(fakeAccountWith());

		await expect
			.element(page.getByLabelText('New dataset name'))
			.toHaveValue('me/llama-chat-traces');
		await expect.element(page.getByText('Create a dataset')).toBeVisible();
	});

	it('shows a few tiles and the rest behind "Show more"', async () => {
		renderDialog(fakeAccountWith('a', 'b', 'c', 'd', 'e'));

		await expect.element(tile('c')).toBeVisible();
		expect(page.getByRole('button', { name: /^d\b/ }).elements()).toHaveLength(0);

		await userEvent.click(page.getByRole('button', { name: 'Show 2 more' }));
		await expect.element(tile('e')).toBeVisible();
	});

	it('picks again when the Hub turns the picked dataset down, instead of keeping a stale pick', async () => {
		const account = fakeAccountWith('alpha', 'beta');

		account.refused.add(`${FAKE_OWNER}/alpha`);
		renderDialog(account);

		await userEvent.click(page.getByRole('button', { exact: true, name: 'Share' }));
		await expect.element(page.getByText('cannot add traces to me/alpha')).toBeVisible();

		// the refused tile is gone and the next one is picked: Share does not aim at it again
		expect(page.getByRole('button', { name: /^alpha\b/ }).elements()).toHaveLength(0);
		await expect.element(tile('beta')).toHaveAttribute('aria-pressed', 'true');

		await userEvent.click(page.getByRole('button', { exact: true, name: 'Share' }));
		await expect.element(page.getByText('Shared to Hugging Face')).toBeVisible();
		expect(account.uploads).toEqual([`${FAKE_OWNER}/alpha`, `${FAKE_OWNER}/beta`]);
	});

	it('picks again for the next account when the signed-in one changes', async () => {
		const account = fakeAccountWith('alpha');

		renderDialog(account);
		await expect.element(tile('alpha')).toHaveAttribute('aria-pressed', 'true');

		// another account signs in, with a dataset of its own, while the dialog is open
		account.user = null;
		account.datasets = [{ id: 'you/zeta', isPrivate: false }];
		account.user = { name: 'you' };

		await expect.element(tile('zeta')).toHaveAttribute('aria-pressed', 'true');
	});
});
