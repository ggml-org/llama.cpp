import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

import { defineConfig, searchForWorkspaceRoot } from 'vite';
import devtoolsJson from 'vite-plugin-devtools-json';
import { storybookTest } from '@storybook/addon-vitest/vitest-plugin';
import { llamaCppBuildPlugin } from './scripts/vite-plugin-llama-cpp-build';
import { playwright } from '@vitest/browser-playwright';

const __dirname = dirname(fileURLToPath(import.meta.url));

const SERVER_ORIGIN = import.meta.env?.VITE_PUBLIC_SERVER_ORIGIN || 'http://localhost:8080';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const browserBaseConfig: any = {
	enabled: true,
	provider: playwright({
		launchOptions: {
			args: ['--no-sandbox']
		}
	}),
	instances: [{ browser: 'chromium' }]
};

export default defineConfig({
	resolve: {
		alias: {
			'katex-fonts': resolve('node_modules/katex/dist/fonts')
		}
	},

	build: {
		assetsInlineLimit: 32000,
		chunkSizeWarningLimit: 3072,
		minify: true
	},

	plugins: [tailwindcss(), sveltekit(), devtoolsJson(), llamaCppBuildPlugin()],

	test: {
		projects: [
			{
				extends: './vite.config.ts',
				test: {
					name: 'client',
					browser: browserBaseConfig,
					include: ['tests/client/**/*.svelte.{test,spec}.{js,ts}'],
					setupFiles: ['./vitest-setup-client.ts']
				}
			},

			{
				extends: './vite.config.ts',
				test: {
					name: 'unit',
					environment: 'node',
					include: ['tests/unit/**/*.{test,spec}.{js,ts}']
				}
			},

			{
				extends: './vite.config.ts',
				test: {
					name: 'ui',
					browser: { ...browserBaseConfig, instances: [{ browser: 'chromium', headless: true }] },
					setupFiles: ['./.storybook/vitest.setup.ts']
				},
				plugins: [
					storybookTest({
						storybookScript: 'pnpm run storybook --no-open'
					})
				]
			}
		]
	},

	server: {
		proxy: {
			'/v1': SERVER_ORIGIN,
			'/props': SERVER_ORIGIN,
			'/models': SERVER_ORIGIN,
			'/tools': SERVER_ORIGIN,
			'/slots': SERVER_ORIGIN,
			'/cors-proxy': SERVER_ORIGIN
		},
		headers: {
			'Cross-Origin-Embedder-Policy': 'require-corp',
			'Cross-Origin-Opener-Policy': 'same-origin'
		},
		fs: {
			allow: [searchForWorkspaceRoot(process.cwd()), resolve(__dirname, 'tests')]
		}
	}
});
