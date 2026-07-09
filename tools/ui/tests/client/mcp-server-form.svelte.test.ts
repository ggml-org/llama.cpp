import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';
import McpServerFormWrapper from './components/McpServerFormWrapper.svelte';

const AUTHORIZATION_HEADER = 'Authorization';
const BEARER_PREFIX = 'Bearer ';
const BEARER_PLACEHOLDER = 'Paste token here';

/**
 * Client-side tests for the McpServerForm bearer UI.
 *
 * The dedicated UI only "owns" Authorization headers that already carry a
 * Bearer scheme (heuristic check on the value). Other Authorization values
 * stay in the KV section so the user can still edit them verbatim. Storage
 * always goes through the same custom-headers slot, so a round-trip via this
 * UI produces exactly one `Authorization: Bearer <token>` entry.
 *
 * Equivalent parser coverage lives in `tests/unit/headers.test.ts`.
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

		await expect.element(bearerInput(screen)).not.toBeInTheDocument();
	});

	it('toggling Authorization shows the bearer input', async () => {
		const screen = await render(McpServerFormWrapper, { headers: '' });

		await screen.getByRole('switch', { name: /authorization/i }).click();

		await expect.element(bearerInput(screen)).toBeVisible();
	});

	it('typing a token writes the Authorization row with the Bearer prefix prepended', async () => {
		const screen = await render(McpServerFormWrapper, { headers: '' });

		await screen.getByRole('switch', { name: /authorization/i }).click();

		const token = 'super-secret';
		await bearerInput(screen).fill(token);

		const expected = JSON.stringify({ [AUTHORIZATION_HEADER]: `${BEARER_PREFIX}${token}` });
		await expect
			.element(capturedHeaders(screen))
			.toHaveAttribute('data-captured-headers', expected);
	});

	it('pre-existing Bearer header pre-fills the bearer input with the token stripped', async () => {
		const existing = JSON.stringify({
			'X-Trace-Id': 'abc',
			[AUTHORIZATION_HEADER]: `${BEARER_PREFIX}preexisting`
		});

		const screen = await render(McpServerFormWrapper, { headers: existing });

		await expect.element(bearerInput(screen)).toBeVisible();
		await expect.element(bearerInput(screen)).toHaveValue('preexisting');
	});

	it('non-Bearer Authorization is ignored by the dedicated UI and stays in the KV section', async () => {
		const existing = JSON.stringify({ [AUTHORIZATION_HEADER]: 'Basic czNjcjpwYXNz' });

		const screen = await render(McpServerFormWrapper, { headers: existing });

		await expect.element(bearerInput(screen)).not.toBeInTheDocument();

		const headerKeyInput = screen.getByPlaceholder('Header name');
		await expect.element(headerKeyInput).toBeVisible();
	});

	it('engaging the token UI replaces a non-Bearer Authorization with the Bearer scheme', async () => {
		const existing = JSON.stringify({ [AUTHORIZATION_HEADER]: 'Basic old' });

		const screen = await render(McpServerFormWrapper, { headers: existing });

		await screen.getByRole('switch', { name: /authorization/i }).click();
		await bearerInput(screen).fill('new');

		const expected = JSON.stringify({ [AUTHORIZATION_HEADER]: `${BEARER_PREFIX}new` });
		await expect
			.element(capturedHeaders(screen))
			.toHaveAttribute('data-captured-headers', expected);
	});

	it('toggling Authorization off with no token drops the Bearer row but keeps non-Bearer schemes', async () => {
		const existing = JSON.stringify({ [AUTHORIZATION_HEADER]: `${BEARER_PREFIX}xyz` });
		const screen = await render(McpServerFormWrapper, { headers: existing });

		await screen.getByRole('switch', { name: /authorization/i }).click();

		await expect.element(capturedHeaders(screen)).toHaveAttribute('data-captured-headers', '');
	});

	it('toggling Authorization off when no Bearer row is present leaves headers untouched', async () => {
		const existing = JSON.stringify({ [AUTHORIZATION_HEADER]: 'Basic czNjcjpwYXNz' });
		const screen = await render(McpServerFormWrapper, { headers: existing });

		await screen.getByRole('switch', { name: /authorization/i }).click();
		await screen.getByRole('switch', { name: /authorization/i }).click();

		await expect
			.element(capturedHeaders(screen))
			.toHaveAttribute('data-captured-headers', existing);
	});

	it('clearing the bearer input drops the Authorization row', async () => {
		const existing = JSON.stringify({ [AUTHORIZATION_HEADER]: `${BEARER_PREFIX}xyz` });
		const screen = await render(McpServerFormWrapper, { headers: existing });

		await bearerInput(screen).fill('');

		await expect.element(capturedHeaders(screen)).toHaveAttribute('data-captured-headers', '');
	});

	it('does not surface Bearer Authorization in the KV section even when pre-existing', async () => {
		const existing = JSON.stringify({ [AUTHORIZATION_HEADER]: `${BEARER_PREFIX}xyz` });
		const screen = await render(McpServerFormWrapper, { headers: existing });

		const headerKeyInput = screen.getByPlaceholder('Header name');
		await expect.element(headerKeyInput).not.toBeInTheDocument();
	});
});
