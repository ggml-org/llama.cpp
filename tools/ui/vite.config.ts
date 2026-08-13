import { buildInfoPlugin } from './scripts/vite-plugin-build-info';
import { nerdamerPlugin } from './scripts/vite-plugin-nerdamer';
import { relativizeBasePlugin } from './scripts/vite-plugin-relativize-base';
import { splashScreenPlugin } from './scripts/vite-plugin-splash-screen';
import { SVELTEKIT_PWA_OPTIONS } from './src/lib/constants/pwa.constants';
import { storybookTest } from '@storybook/addon-vitest/vitest-plugin';
import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { SvelteKitPWA } from '@vite-pwa/sveltekit';
import { playwright } from '@vitest/browser-playwright';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';
import { defineConfig, searchForWorkspaceRoot } from 'vite';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SERVER_ORIGIN = import.meta.env?.VITE_PUBLIC_SERVER_ORIGIN || 'http://localhost:8080';
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const browserBaseConfig: any = {
	enabled: true,
	instances: [{ browser: 'chromium' }],
	provider: playwright({
		launchOptions: {
			args: ['--no-sandbox']
		}
	})
};

export default defineConfig({
	build: {
		assetsInlineLimit: 32000,
		chunkSizeWarningLimit: 3072,
		minify: true
	},

	plugins: [
		tailwindcss(),
		sveltekit(),
		SvelteKitPWA(SVELTEKIT_PWA_OPTIONS),
		splashScreenPlugin(),
		buildInfoPlugin(),
		nerdamerPlugin(),
		relativizeBasePlugin()
	],

	resolve: {
		alias: {
			'katex-fonts': resolve('node_modules/katex/dist/fonts')
		}
	},

	server: {
		fs: {
			allow: [searchForWorkspaceRoot(process.cwd()), resolve(__dirname, 'tests')]
		},
		headers: {
			'Cross-Origin-Embedder-Policy': 'require-corp',
			'Cross-Origin-Opener-Policy': 'same-origin'
		},
		proxy: {
			'/cors-proxy': SERVER_ORIGIN,
			'/models': SERVER_ORIGIN,
			'/props': SERVER_ORIGIN,
			'/slots': SERVER_ORIGIN,
			'/tools': SERVER_ORIGIN,
			'/v1': SERVER_ORIGIN
		}
	},

	test: {
		projects: [
			{
				extends: './vite.config.ts',
				test: {
					browser: browserBaseConfig,
					include: ['tests/client/**/*.svelte.{test,spec}.{js,ts}'],
					name: 'client',
					setupFiles: ['./vitest-setup-client.ts']
				}
			},

			{
				extends: './vite.config.ts',
				test: {
					environment: 'node',
					include: ['tests/unit/**/*.{test,spec}.{js,ts}'],
					name: 'unit'
				}
			},

			{
				extends: './vite.config.ts',
				plugins: [
					storybookTest({
						storybookScript: 'pnpm run storybook --no-open'
					})
				],
				test: {
					browser: { ...browserBaseConfig, instances: [{ browser: 'chromium', headless: true }] },
					name: 'ui'
				}
			}
		]
	}
});
