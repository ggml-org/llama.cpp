import { defineConfig } from '@playwright/test';

export default defineConfig({
	webServer: {
		command:
			'npm run build && gzip -dc ../public/index.html.gz > ../public/index.html && http-server ../public -p 8181',
		port: 8181,
		timeout: 120000,
		reuseExistingServer: false
	},
	testDir: 'tests/e2e'
});
