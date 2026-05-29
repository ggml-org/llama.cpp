import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { SvelteKitPWA } from '@vite-pwa/sveltekit';
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

	plugins: [
		tailwindcss(),
		sveltekit(),
		SvelteKitPWA({
			// Strategy: generateSW - the plugin generates a service worker automatically
			// using Workbox. For a custom SW, use 'injectManifest' instead.
			// Manifest configuration
			manifest: {
				name: 'llama-ui',
				short_name: 'llama-ui',
				description: 'Local AI chat interface powered by llama.cpp',
				start_url: './',
				display: 'standalone',
				background_color: 'white',
				theme_color: 'white',
				icons: [
					{
						src: 'pwa-64x64.png',
						sizes: '64x64',
						type: 'image/png'
					},
					{
						src: 'pwa-192x192.png',
						sizes: '192x192',
						type: 'image/png'
					},
					{
						src: 'pwa-512x512.png',
						sizes: '512x512',
						type: 'image/png',
						purpose: 'any'
					},
					{
						src: 'maskable-icon-512x512.png',
						sizes: '512x512',
						type: 'image/png',
						purpose: 'maskable'
					}
				]
			},

			// Workbox configuration for generateSW strategy
			workbox: {
				// Match all static assets in the build output.
				// Uses '**/' because SvelteKit outputs files under _app/immutable/
				// subdirectories; the llama-cpp-build plugin flattens them after
				// the SW is generated.
				globPatterns: ['**/*.{js,css,html,ico,svg,png,webp,woff,woff2,json,webmanifest}'],

				maximumFileSizeToCacheInBytes: 6 * 1024 * 1024, // 6 MB — bundle.js is ~5.3 MB

				// Runtime caching for API calls - use NetworkFirst so APIs are always fresh
				runtimeCaching: [
					{
						urlPattern: /\/v1\/.*/,
						handler: 'NetworkFirst',
						options: {
							cacheName: 'api-cache',
							expiration: {
								maxEntries: 50,
								maxAgeSeconds: 60 * 60 * 24 // 24 hours
							}
						}
					},
					{
						urlPattern: /\/(health|props|models|tools|slots|cors-proxy).*/,
						handler: 'NetworkFirst',
						options: {
							cacheName: 'api-cache',
							expiration: {
								maxEntries: 50,
								maxAgeSeconds: 60 * 60 * 24
							}
						}
					}
				]
			},

			// Dev options - enable SW in development mode for testing
			devOptions: {
				enabled: true,
				suppressWarnings: true,
				// Use '/' to match production SW behaviour (navigateFallback defaults to
				// the configured base path, which is '/' for this SPA).
				navigateFallback: '/'
			},

			// SvelteKit-specific options
			kit: {
				// Include version file for proper cache invalidation
				includeVersionFile: true
			}
		}),
		devtoolsJson(),
		llamaCppBuildPlugin()
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
