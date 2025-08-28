import { defineConfig } from '@playwright/test';

export default defineConfig({
	webServer: {
		command: 'npm run build && npx http-server ../public -p 8181',
		port: 8181
	},
	testDir: 'e2e'
});
