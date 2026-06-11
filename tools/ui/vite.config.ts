import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { SvelteKitPWA } from '@vite-pwa/sveltekit';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

import { defineConfig, searchForWorkspaceRoot } from 'vite';
import { storybookTest } from '@storybook/addon-vitest/vitest-plugin';
import { splashScreenPlugin } from './scripts/vite-plugin-splash-screen';
import { buildInfoPlugin } from './scripts/vite-plugin-build-info';
import { relativizeBasePlugin } from './scripts/vite-plugin-relativize-base';
import { playwright } from '@vitest/browser-playwright';
import {
	PWA_MANIFEST,
	CACHE_SETTINGS,
	GLOB_PATTERNS,
	RUNTIME_CACHING,
	API_CACHING_PATTERNS,
	PWA_KIT_OPTIONS
} from './src/lib/constants/pwa';

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

	plugins: [
		tailwindcss(),
		sveltekit(),
		SvelteKitPWA({
			// Strategy: generateSW - the plugin generates a service worker automatically
			// using Workbox. For a custom SW, use 'injectManifest' instead.
			// Manifest configuration
			manifest: PWA_MANIFEST,

			// Workbox configuration for generateSW strategy
			workbox: {
				// Match all static assets in the build output.
				// Uses '**/' because SvelteKit outputs files under _app/immutable/
				// subdirectories.
				globPatterns: GLOB_PATTERNS,

				maximumFileSizeToCacheInBytes: CACHE_SETTINGS.MAX_FILE_SIZE_BYTES,

				// Runtime caching for API calls - use NetworkFirst so APIs are always fresh
				runtimeCaching: [
					{
						urlPattern: API_CACHING_PATTERNS.V1_API,
						handler: RUNTIME_CACHING.HANDLER,
						options: {
							cacheName: RUNTIME_CACHING.CACHE_NAME,
							expiration: {
								maxEntries: CACHE_SETTINGS.API_CACHE_MAX_ENTRIES,
								maxAgeSeconds: CACHE_SETTINGS.API_CACHE_MAX_AGE_SECONDS
							}
						}
					},
					{
						urlPattern: API_CACHING_PATTERNS.STATIC_API,
						handler: RUNTIME_CACHING.HANDLER,
						options: {
							cacheName: RUNTIME_CACHING.CACHE_NAME,
							expiration: {
								maxEntries: CACHE_SETTINGS.API_CACHE_MAX_ENTRIES,
								maxAgeSeconds: CACHE_SETTINGS.API_CACHE_MAX_AGE_SECONDS
							}
						}
					}
				]
			},

			devOptions: {
				enabled: true,
				suppressWarnings: true,
				// Use PWA_KIT_OPTIONS.NAVIGATE_FALLBACK to match production SW behaviour
				// (navigateFallback defaults to the configured base path, which is '/' for this SPA).
				navigateFallback: PWA_KIT_OPTIONS.NAVIGATE_FALLBACK
			},

			// SvelteKit-specific options
			kit: {
				// Include version file for proper cache invalidation
				includeVersionFile: true
			}
		}),
		splashScreenPlugin(),
		buildInfoPlugin(),
		relativizeBasePlugin()
	],

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
