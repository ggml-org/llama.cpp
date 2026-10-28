import { DEMO_MCP_PORT } from './fixtures/demo-config.ts';
import { defineConfig, devices } from '@playwright/test';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

const ENV_FILE = resolve(import.meta.dirname, '../.env');

if (existsSync(ENV_FILE)) process.loadEnvFile(ENV_FILE);

/**
 * Demo video recordings.
 *
 * Env (also read from tools/ui/.env):
 *   DEMO_UI_DIR         tools/ui checkout to record (default: this one), e.g. a PR worktree
 *   DEMO_UI_PORT        port for that checkout's vite dev server (default: 5180)
 *   DEMO_SERVER_ORIGIN  llama-server the UI proxies to (default: http://localhost:9931)
 *   DEMO_LLAMA_SERVER   llama-server binary started when none runs there (default: build/bin)
 *   DEMO_COLOR_SCHEME   light | dark (default: light)
 */

// Playwright clears outputDir when a run starts, so concurrent recordings each get
// their own; workers inherit the env, so they resolve the same directory
process.env.DEMO_RUN_ID ??= String(process.pid);

const UI_DIR = resolve(process.env.DEMO_UI_DIR ?? resolve(import.meta.dirname, '..'));
const UI_PORT = Number(process.env.DEMO_UI_PORT ?? 5180);
const SERVER_ORIGIN = process.env.DEMO_SERVER_ORIGIN ?? 'http://localhost:9931';
const SERVER_PORT = Number(new URL(SERVER_ORIGIN).port || 80);
const LLAMA_SERVER =
	process.env.DEMO_LLAMA_SERVER ?? resolve(import.meta.dirname, '../../../build/bin/llama-server');

export const VIEWPORT = { height: 900, width: 1440 };

export default defineConfig({
	expect: {
		timeout: 10000
	},
	fullyParallel: false,
	outputDir: `test-results/${process.env.DEMO_RUN_ID}`,
	projects: [
		{
			name: 'chromium',
			use: { ...devices['Desktop Chrome'], viewport: VIEWPORT }
		}
	],
	reporter: 'line',
	retries: 0,
	testDir: '.',
	testMatch: ['**/*.demo.ts'],
	timeout: 120000,
	use: {
		actionTimeout: 10000,
		baseURL: `http://localhost:${UI_PORT}`,
		colorScheme: process.env.DEMO_COLOR_SCHEME === 'dark' ? 'dark' : 'light',
		video: { mode: 'on', size: VIEWPORT },
		viewport: VIEWPORT
	},
	webServer: [
		{
			command: `node ${resolve(import.meta.dirname, 'fixtures/mcp-servers.ts')}`,
			reuseExistingServer: true,
			url: `http://localhost:${DEMO_MCP_PORT}/health`
		},
		{
			command: `${LLAMA_SERVER} --host 127.0.0.1 --port ${SERVER_PORT} --tools all`,
			reuseExistingServer: true,
			timeout: 60000,
			url: `${SERVER_ORIGIN}/health`
		},
		{
			command: `npx vite dev --port ${UI_PORT} --strictPort`,
			cwd: UI_DIR,
			env: { VITE_PUBLIC_SERVER_ORIGIN: SERVER_ORIGIN },
			reuseExistingServer: true,
			timeout: 120000,
			url: `http://localhost:${UI_PORT}`
		}
	],
	workers: 1
});
