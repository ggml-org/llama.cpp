import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';
import McpServerFormWrapper from './components/McpServerFormWrapper.svelte';

const AUTHORIZATION_HEADER = 'Authorization';
const BEARER_PREFIX = 'Bearer ';
const BEARER_PLACEHOLDER = 'Paste token here';

/**
 * Client-side tests for the McpServerForm bearer UI added on this branch.
 *
 * The form is the only place where the recommendable `Authorization` header
 * can be written without going through the KV section, so we cover its
 * round-trip here end-to-end: empty state, toggle on, type a token, persist
 * on close, fix non-Bearer values, etc. Equivalent parser coverage lives
 * in `tests/unit/headers.test.ts`.
 */
describe('McpServerForm - Authorization / bearer UI', () => {
	function bearerInput(screen: Awaited<ReturnType<typeof render>>) {
		return screen.locator.getByPlaceholder(BEARER_PLACEHOLDER);
	}

	function capturedHeaders(screen: Awaited<ReturnType<typeof render>>) {
		return screen.getByTestId('captured-headers');
	}

	it('mounts with the bearer input hidden when no auth header is present', async () => {
		const screen = await render(McpServerFormWrapper, { headers: '' });

		await expect.element(screen.getByRole('textbox', { name: /server url/i })).toBeVisible();

		// No row matches because #if showAuthorization gates the bearer input.
		await expect.element(bearerInput(screen)).not.toBeInTheDocument();
	});

	it('toggling Authorization shows the bearer input', async () => {
		const screen = await render(McpServerFormWrapper, { headers: '' });

		await screen.getByRole('switch', { name: /authorization/i }).click();

		await expect.element(bearerInput(screen)).toBeVisible();
	});

	it('typing a bearer token writes Authorization row to the headers prop', async () => {
		const screen = await render(McpServerFormWrapper, { headers: '' });

		await screen.getByRole('switch', { name: /authorization/i }).click();

		const token = 'super-secret';
		await bearerInput(screen).fill(token);

		const expected = JSON.stringify({ [AUTHORIZATION_HEADER]: `${BEARER_PREFIX}${token}` });
		await expect
			.element(capturedHeaders(screen))
			.toHaveAttribute('data-captured-headers', expected);
	});

	it('pre-existing Bearer header pre-fills the bearer input', async () => {
		const existing = JSON.stringify({
			'X-Trace-Id': 'abc',
			[AUTHORIZATION_HEADER]: `${BEARER_PREFIX}preexisting`
		});

		const screen = await render(McpServerFormWrapper, { headers: existing });

		await expect.element(bearerInput(screen)).toBeVisible();
		await expect.element(bearerInput(screen)).toHaveValue('preexisting');
	});

	it('clearing the bearer input drops the Authorization row', async () => {
		const existing = JSON.stringify({ [AUTHORIZATION_HEADER]: `${BEARER_PREFIX}xyz` });
		const screen = await render(McpServerFormWrapper, { headers: existing });

		await bearerInput(screen).fill('');

		await expect.element(capturedHeaders(screen)).toHaveAttribute('data-captured-headers', '');
	});

	it('toggling Authentication off removes the bearer row from headers', async () => {
		const existing = JSON.stringify({ [AUTHORIZATION_HEADER]: `${BEARER_PREFIX}xyz` });
		const screen = await render(McpServerFormWrapper, { headers: existing });

		await screen.getByRole('switch', { name: /authorization/i }).click();

		await expect.element(capturedHeaders(screen)).toHaveAttribute('data-captured-headers', '');
	});

	it('does not surface Authorization in the KV section even when pre-existing', async () => {
		const existing = JSON.stringify({ [AUTHORIZATION_HEADER]: `${BEARER_PREFIX}xyz` });
		const screen = await render(McpServerFormWrapper, { headers: existing });

		// KV rows are keyed by their placeholder; if Authorization leaked
		// into the KV section we'd see a "Header name" input visible.
		const headerKeyInput = screen.getByPlaceholder('Header name');
		await expect.element(headerKeyInput).not.toBeInTheDocument();
	});
});
