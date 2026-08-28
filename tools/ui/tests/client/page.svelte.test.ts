import TestWrapper from './components/TestWrapper.svelte';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

describe('/+page.svelte', () => {
	it('should render page without throwing', async () => {
		// Basic smoke test - page should render without throwing errors.
		// API calls are mocked in vitest-setup-client.ts.
		await render(TestWrapper);
		expect(true).toBe(true);
	});
});
