/**
 * Centralized PWA constants to avoid magic strings, regexes, and duplicated
 * definitions across the codebase.
 */

import { APP_NAME } from './app';

export const MEDIA_QUERIES = {
	PREFERS_DARK: '(prefers-color-scheme: dark)',
	PREFERS_LIGHT: '(prefers-color-scheme: light)',
	DISPLAY_MODE_STANDALONE: '(display-mode: standalone)'
} as const;

export const THEME_COLORS = {
	LIGHT: '#ffffff',
	DARK: '#0d0d0d',
	ACCENT_BLUE: '#2563eb',
	ACCENT_BLUE_HOVER: '#1d4ed8',
	BACKGROUND_LIGHT: 'white',
	BACKGROUND_DARK: '#111111',
	TITLE_UPDATE_ALERT: {
		BORDER_LIGHT: 'zinc-200',
		BORDER_DARK: 'zinc-700',
		BG_LIGHT: 'white',
		BG_DARK: 'zinc-800',
		TEXT_LIGHT: 'zinc-500',
		TEXT_DARK: 'zinc-400'
	}
} as const;

export const FAVICON_PATHS = {
	ICO_LIGHT: 'favicon.ico',
	ICO_DARK: 'favicon-dark.ico',
	SVG_LIGHT: 'favicon.svg',
	SVG_DARK: 'favicon-dark.svg'
} as const;

// Substituted for `currentColor` in src/lib/assets/logo.svg when generating
// the light/dark static sources consumed by the PWA asset generator.
export const FAVICON_COLORS = {
	LIGHT: '#111111',
	DARK: '#fafafa'
} as const;

export const FAVICON_SELECTORS = {
	ICO_48X48: 'link[rel="icon"][sizes="48x48"]',
	SVG_ANY: 'link[rel="icon"][type="image/svg+xml"]'
} as const;

export const APPLE_ASSETS = {
	TOUCH_ICON: 'apple-touch-icon-180x180.png'
} as const;

export const PWA_MANIFEST = {
	name: APP_NAME,
	short_name: APP_NAME,
	description: 'Local AI chat interface powered by llama.cpp',
	start_url: './',
	display: 'standalone' as const,
	background_color: THEME_COLORS.BACKGROUND_LIGHT,
	theme_color: THEME_COLORS.BACKGROUND_LIGHT,
	icons: [
		{ src: 'pwa-64x64.png', sizes: '64x64', type: 'image/png' },
		{ src: 'pwa-192x192.png', sizes: '192x192', type: 'image/png' },
		{ src: 'pwa-512x512.png', sizes: '512x512', type: 'image/png', purpose: 'any' as const },
		{
			src: 'maskable-icon-512x512.png',
			sizes: '512x512',
			type: 'image/png',
			purpose: 'maskable' as const
		}
	]
};

export const PWA_ICON_PATHS = {
	PWA_64: '/pwa-64x64.png',
	PWA_192: '/pwa-192x192.png',
	PWA_512: '/pwa-512x512.png',
	MASKABLE_512: '/maskable-icon-512x512.png'
} as const;

/** Apple device dimensions (logical points) and DPR, from Apple HIG. */
export const APPLE_DEVICES = {
	// iPhones (DPR 3)
	'1170x2532': { width: 390, height: 844, dpr: 3 }, // iPhone 13, 15
	'1179x2556': { width: 393, height: 852, dpr: 3 }, // iPhone 14, 15 Pro, 16
	'1206x2622': { width: 402, height: 874, dpr: 3 }, // iPhone 16 Plus, 16e
	'1284x2778': { width: 428, height: 926, dpr: 3 }, // iPhone 15 Plus
	'1290x2796': { width: 430, height: 932, dpr: 3 }, // iPhone 15 Pro Max, 16 Pro
	'1320x2868': { width: 440, height: 956, dpr: 3 }, // iPhone 16 Pro Max
	'750x1334': { width: 375, height: 667, dpr: 2 }, // iPhone 6/7/8, 14
	'640x1136': { width: 320, height: 568, dpr: 2 }, // iPhone 6/7/8 Plus
	// iPads (DPR 2)
	'1668x2388': { width: 834, height: 1194, dpr: 2 }, // iPad Air 11", iPad 11"
	'2048x2732': { width: 1024, height: 1366, dpr: 2 }, // iPad Pro 12.9"
	'1640x2360': { width: 820, height: 1180, dpr: 2 }, // iPad Air 10.9"
	'1032x1376': { width: 1032, height: 1376, dpr: 2 }, // iPad Air 13"
	'744x1133': { width: 376, height: 573, dpr: 2 } // iPad mini 8.3"
} as const;

export type AppleDeviceKey = keyof typeof APPLE_DEVICES;

export const PWA_FILE_PATHS = {
	MANIFEST: '/manifest.webmanifest',
	SERVICE_WORKER: '/sw.js',
	VERSION: '/version.json',
	WORKBOX: '/workbox-<hash>.js'
} as const;

// Used by the server middleware to skip API key validation.
// Keep in sync with tools/server/server-http.cpp public_endpoints list.

export const PUBLIC_ENDPOINTS = [
	'/health',
	'/v1/health',
	'/models',
	'/v1/models',
	'/props',
	'/metrics',
	'/',
	'/index.html',

	'/favicon.ico',
	'/favicon-dark.ico',
	'/favicon.svg',
	'/favicon-dark.svg',
	'/pwa-64x64.png',
	'/pwa-192x192.png',
	'/pwa-512x512.png',
	'/maskable-icon-512x512.png',
	'/apple-touch-icon-180x180.png',
	'/apple-splash-portrait-640x1136.png',
	'/apple-splash-landscape-640x1136.png',
	'/apple-splash-portrait-750x1334.png',
	'/apple-splash-landscape-750x1334.png',
	'/apple-splash-portrait-1170x2532.png',
	'/apple-splash-landscape-1170x2532.png',
	'/apple-splash-portrait-1179x2556.png',
	'/apple-splash-landscape-1179x2556.png',
	'/apple-splash-portrait-1206x2622.png',
	'/apple-splash-landscape-1206x2622.png',
	'/apple-splash-portrait-1284x2778.png',
	'/apple-splash-landscape-1284x2778.png',
	'/apple-splash-portrait-1290x2796.png',
	'/apple-splash-landscape-1290x2796.png',
	'/apple-splash-portrait-1320x2868.png',
	'/apple-splash-landscape-1320x2868.png',
	'/apple-splash-portrait-1488x2266.png',
	'/apple-splash-landscape-1488x2266.png',
	'/apple-splash-portrait-1640x2360.png',
	'/apple-splash-landscape-1640x2360.png',
	'/apple-splash-portrait-1668x2388.png',
	'/apple-splash-landscape-1668x2388.png',
	'/apple-splash-portrait-2048x2732.png',
	'/apple-splash-landscape-2048x2732.png',
	'/apple-splash-portrait-dark-640x1136.png',
	'/apple-splash-landscape-dark-640x1136.png',
	'/apple-splash-portrait-dark-750x1334.png',
	'/apple-splash-landscape-dark-750x1334.png',
	'/apple-splash-portrait-dark-1170x2532.png',
	'/apple-splash-landscape-dark-1170x2532.png',
	'/apple-splash-portrait-dark-1179x2556.png',
	'/apple-splash-landscape-dark-1179x2556.png',
	'/apple-splash-portrait-dark-1206x2622.png',
	'/apple-splash-landscape-dark-1206x2622.png',
	'/apple-splash-portrait-dark-1284x2778.png',
	'/apple-splash-landscape-dark-1284x2778.png',
	'/apple-splash-portrait-dark-1290x2796.png',
	'/apple-splash-landscape-dark-1290x2796.png',
	'/apple-splash-portrait-dark-1320x2868.png',
	'/apple-splash-landscape-dark-1320x2868.png',
	'/apple-splash-portrait-dark-1488x2266.png',
	'/apple-splash-landscape-dark-1488x2266.png',
	'/apple-splash-portrait-dark-1640x2360.png',
	'/apple-splash-landscape-dark-1640x2360.png',
	'/apple-splash-portrait-dark-1668x2388.png',
	'/apple-splash-landscape-dark-1668x2388.png',
	'/apple-splash-portrait-dark-2048x2732.png',
	'/apple-splash-landscape-dark-2048x2732.png',
	'/manifest.webmanifest',
	'/sw.js',
	'/version.json',
	'/workbox-<hash>.js'
] as const;
export const BUILD_CONFIG = {
	OUTPUT_DIR: './dist',
	GUIDE_COMMENT: `
<!--
  This is a static build of the frontend.
  It is automatically generated by the build process.
  Do not edit this file directly.
  To make changes, refer to the "Web UI" section in the README.
-->
`.trim()
} as const;

export const REGEX_PATTERNS = {
	SPLASH_FILE: /^apple-splash-(portrait|landscape)-(dark-)?(\d+)x(\d+)\.png$/,
	HEAD_CLOSE: /\t*<\/head>/
} as const;

// Device names used by @vite-pwa/assets-generator for splash screen generation.
// Keep in sync with pwa-assets.config.ts.
export const PWA_GENERATOR_DEVICES = [
	'iPhone 13',
	'iPhone 13 Pro',
	'iPhone 13 Pro Max',
	'iPhone 14',
	'iPhone 14 Plus',
	'iPhone 14 Pro',
	'iPhone 14 Pro Max',
	'iPhone 15',
	'iPhone 15 Plus',
	'iPhone 15 Pro',
	'iPhone 15 Pro Max',
	'iPhone 16',
	'iPhone 16 Plus',
	'iPhone 16 Pro',
	'iPhone 16 Pro Max',
	'iPhone 16e',
	'iPhone SE 4"',
	'iPhone SE 4.7"',
	'iPad 11"',
	'iPad Air 10.9"',
	'iPad Air 11"',
	'iPad Air 13"',
	'iPad Pro 11"',
	'iPad Pro 12.9"',
	'iPad mini 8.3"'
] as const;

// PWA assets generator configuration — used by pwa-assets.config.ts
// FAVICON_PADDING: fraction (0..1) of the icon reserved as equal margin on
// each side. Applied to icon PNG/ICO outputs by @vite-pwa/assets-generator and
// post-processed into the static favicon.svg so the in-app logo (which reads
// src/lib/assets/logo.svg directly) is unaffected.
export const PWA_ASSET_GENERATOR = {
	LINK_PRESET: '2023',
	FAVICON_PADDING: 0.04,
	SPLASH_PADDING: 0.75,
	FIT_MODE: 'contain',
	ADD_MEDIA_SCREEN: true,
	BASE_PATH: './',
	XHTML: false,
	PNG_COMPRESSION_LEVEL: 9,
	PNG_QUALITY: 60,
	DARK_PREFIX: 'dark-'
} as const;

export const CACHE_SETTINGS = {
	IMMUTABLE_MAX_AGE_SECONDS: 31536000,
	API_CACHE_MAX_AGE_SECONDS: 60 * 60 * 24,
	API_CACHE_MAX_ENTRIES: 50,
	MAX_FILE_SIZE_BYTES: 10 * 1024 * 1024
} as const;

export const GLOB_PATTERNS: string[] = [
	'**/*.{js,css,html,ico,svg,png,webp,woff,woff2,json,webmanifest}'
];

export const SW_CONFIG = {
	CHECK_INTERVAL_MS: 60000,
	UPDATE_FETCH_OPTIONS: {
		CACHE: 'no-store',
		HEADERS: {
			CACHE: 'no-store',
			CACHE_CONTROL: 'no-cache'
		}
	}
} as const;

// Runtime caching configuration for Workbox
export const RUNTIME_CACHING = {
	HANDLER: 'NetworkFirst',
	CACHE_NAME: 'api-cache'
} as const;

// Workbox runtime caching patterns
export const API_CACHING_PATTERNS = {
	V1_API: /^\/v1\/.*/,
	STATIC_API: /^\/(health|props|models|tools|slots|cors-proxy).*/
} as const;

// SvelteKit PWA plugin options
export const PWA_KIT_OPTIONS = {} as const;

export const APPLE_META_TAGS = {
	MOBILE_WEB_APP_CAPABLE: { name: 'apple-mobile-web-app-capable', content: 'yes' },
	STATUS_BAR_STYLE: { name: 'apple-mobile-web-app-status-bar-style', content: 'black-translucent' },
	MOBILE_WEB_APP_TITLE: { name: 'apple-mobile-web-app-title' }
} as const;

// Splash screen HTML link tag prefix used by generateSplashScreenLinks
export const SPLASH_LINK = {
	HTML: '<link rel="apple-touch-startup-image"',
	DARK_MEDIA_SUFFIX: ' and (prefers-color-scheme: dark)'
} as const;

// SvelteKit PWA plugin configuration — used by @vite.config.ts
import type { SvelteKitPWAOptions } from '@vite-pwa/sveltekit';

export const SVELTEKIT_PWA_OPTIONS: SvelteKitPWAOptions = {
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

		// Prevent @vite-pwa/sveltekit from auto-adding a NavigationRoute by
		// setting navigateFallback to empty string. This keeps the service
		// worker from intercepting direct browser navigation to server API
		// endpoints (e.g. /slots, /models, /v1/models) which should return
		// JSON, not the SPA HTML shell. The server's own static-file fallback
		// handles non-API navigation to index.html for the SPA router.
		navigateFallback: '',

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
		suppressWarnings: true
	},

	// SvelteKit-specific options
	kit: {
		// Include version file for proper cache invalidation
		includeVersionFile: true
	}
};
