import { defineConfig } from '@playwright/test';

export default defineConfig({
	webServer: {
		command: 'npm run build && http-server ../../build/tools/ui/dist -p 8181',
		port: 8181,
		timeout: 120000,
		reuseExistingServer: false
	},
	testDir: 'tests/e2e'
});
