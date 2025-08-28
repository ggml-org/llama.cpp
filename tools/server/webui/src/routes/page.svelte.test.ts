// import { page } from '@vitest/browser/context';
import {
	describe,
	// expect,
	it
} from 'vitest';
import { render } from 'vitest-browser-svelte';
import Page from './+page.svelte';

describe('/+page.svelte', () => {
	it('should render page', async () => {
		render(Page);

		// const heading = page.getByText('llama.cpp');
		// await expect.element(heading).toBeInTheDocument();
	});
});
