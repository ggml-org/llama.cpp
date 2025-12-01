import { describe, it, expect } from 'vitest';
import { render } from 'vitest-browser-svelte';
import TestWrapper from './components/TestWrapper.svelte';

describe('/+page.svelte', () => {
	it('should render page without throwing', async () => {
		// Basic smoke test - page should render without throwing errors
		// API calls will fail in test environment but component should still mount
		expect(() => render(TestWrapper)).not.toThrow();
	});
});
