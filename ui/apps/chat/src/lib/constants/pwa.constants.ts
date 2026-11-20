/**
 * Centralized PWA constants to avoid magic strings, regexes, and duplicated
 * definitions across the codebase.
 */

import { APP_NAME } from './app.constants';

export const MEDIA_QUERIES = {
	COARSE_POINTER: '(pointer: coarse)',
	DISPLAY_MODE_STANDALONE: '(display-mode: standalone)',
	PREFERS_DARK: '(prefers-color-scheme: dark)',
	PREFERS_LIGHT: '(prefers-color-scheme: light)'
} as const;

export const THEME_COLORS = {
	ACCENT_BLUE: '#2563eb',
	ACCENT_BLUE_HOVER: '#1d4ed8',
	BACKGROUND_DARK: '#111111',
	BACKGROUND_LIGHT: 'white',
	DARK: '#0d0d0d',
	LIGHT: '#ffffff',
	TITLE_UPDATE_ALERT: {
		BG_DARK: 'zinc-800',
		BG_LIGHT: 'white',
		BORDER_DARK: 'zinc-700',
		BORDER_LIGHT: 'zinc-200',
		TEXT_DARK: 'zinc-400',
		TEXT_LIGHT: 'zinc-500'
	}
} as const;

export const FAVICON_PATHS = {
	ICO_DARK: 'favicon-dark.ico',
	ICO_LIGHT: 'favicon.ico',
	SVG_DARK: 'favicon-dark.svg',
	SVG_LIGHT: 'favicon.svg'
} as const;

export const FAVICON_SELECTORS = {
	ICO_48X48: 'link[rel="icon"][sizes="48x48"]',
	SVG_ANY: 'link[rel="icon"][type="image/svg+xml"]'
} as const;

export const APPLE_ASSETS = {
	TOUCH_ICON: 'apple-touch-icon-180x180.png'
} as const;

export const PWA_MANIFEST = {
	background_color: THEME_COLORS.BACKGROUND_LIGHT,
	description: 'Local AI chat interface powered by llama.cpp',
	display: 'standalone' as const,
	icons: [
		{ sizes: '64x64', src: 'pwa-64x64.png', type: 'image/png' },
		{ sizes: '192x192', src: 'pwa-192x192.png', type: 'image/png' },
		{ purpose: 'any' as const, sizes: '512x512', src: 'pwa-512x512.png', type: 'image/png' },
		{
			purpose: 'maskable' as const,
			sizes: '512x512',
			src: 'maskable-icon-512x512.png',
			type: 'image/png'
		}
	],
	lang: 'en',
	name: APP_NAME,
	scope: './',
	short_name: APP_NAME,
	start_url: './',
	theme_color: THEME_COLORS.BACKGROUND_LIGHT
};

export const PWA_ICON_PATHS = {
	MASKABLE_512: '/maskable-icon-512x512.png',
	PWA_64: '/pwa-64x64.png',
	PWA_192: '/pwa-192x192.png',
	PWA_512: '/pwa-512x512.png'
} as const;

/** Apple device dimensions (logical points) and DPR, from Apple HIG. */
export const APPLE_DEVICES = {
	'640x1136': { dpr: 2, height: 568, width: 320 }, // iPhone 6/7/8 Plus
	'744x1133': { dpr: 2, height: 573, width: 376 }, // iPad mini 8.3"
	'750x1334': { dpr: 2, height: 667, width: 375 }, // iPhone 6/7/8, 14
	'1032x1376': { dpr: 2, height: 1376, width: 1032 }, // iPad Air 13"
	// iPhones (DPR 3)
	'1170x2532': { dpr: 3, height: 844, width: 390 }, // iPhone 13, 15
	'1179x2556': { dpr: 3, height: 852, width: 393 }, // iPhone 14, 15 Pro, 16
	'1206x2622': { dpr: 3, height: 874, width: 402 }, // iPhone 16 Plus, 16e
	'1284x2778': { dpr: 3, height: 926, width: 428 }, // iPhone 15 Plus
	'1290x2796': { dpr: 3, height: 932, width: 430 }, // iPhone 15 Pro Max, 16 Pro
	'1320x2868': { dpr: 3, height: 956, width: 440 }, // iPhone 16 Pro Max
	'1640x2360': { dpr: 2, height: 1180, width: 820 }, // iPad Air 10.9"
	// iPads (DPR 2)
	'1668x2388': { dpr: 2, height: 1194, width: 834 }, // iPad Air 11", iPad 11"
	'2048x2732': { dpr: 2, height: 1366, width: 1024 } // iPad Pro 12.9"
} as const;

export type AppleDeviceKey = keyof typeof APPLE_DEVICES;

export const PWA_FILE_PATHS = {
	LEGACY_SERVICE_WORKER: '/sw.js',
	MANIFEST: '/manifest.webmanifest',
	SERVICE_WORKER: '/service-worker.js'
} as const;

/** The Vite plugin that serves manifest.webmanifest in dev and emits it with the build */
export const PWA_MANIFEST_PLUGIN = {
	/** Media type of a web app manifest (W3C Web App Manifest) */
	CONTENT_TYPE: 'application/manifest+json',
	/** Vite environment that builds the browser bundle, the one the manifest goes with */
	ENVIRONMENT: 'client',
	NAME: 'llamacpp:pwa-manifest'
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
	'/service-worker.js',
	'/sw.js',
	'/version.json'
] as const;

export const BUILD_CONFIG = {
	GUIDE_COMMENT: `
<!--
  This is a static build of the frontend.
  It is automatically generated by the build process.
  Do not edit this file directly.
  To make changes, refer to the "Llama Chat" section in the README.
-->
`.trim(),
	OUTPUT_DIR: './dist'
} as const;

export const REGEX_PATTERNS = {
	HEAD_CLOSE: /\t*<\/head>/,
	SPLASH_FILE: /^apple-splash-(portrait|landscape)-(dark-)?(\d+)x(\d+)\.png$/
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
// each side of the icon PNG/ICO outputs. 0, because llama-brand's favicons
// already carry their margin and optical centring.
export const PWA_ASSET_GENERATOR = {
	ADD_MEDIA_SCREEN: true,
	BASE_PATH: './',
	DARK_PREFIX: 'dark-',
	FAVICON_PADDING: 0,
	FIT_MODE: 'contain',
	LINK_PRESET: '2023',
	PNG_COMPRESSION_LEVEL: 9,
	PNG_QUALITY: 60,
	SPLASH_PADDING: 0.75,
	XHTML: false
} as const;

export const SW_CONFIG = {
	/** How long the update waits for the new worker to install and activate before reloading */
	ACTIVATE_TIMEOUT_MS: 10000,
	CHECK_INTERVAL_MS: 60000,
	UPDATE_FETCH_OPTIONS: {
		CACHE: 'no-store',
		HEADERS: {
			CACHE: 'no-store',
			CACHE_CONTROL: 'no-cache'
		}
	}
} as const;

export const APPLE_META_TAGS = {
	MOBILE_WEB_APP_CAPABLE: { content: 'yes', name: 'apple-mobile-web-app-capable' },
	MOBILE_WEB_APP_TITLE: { name: 'apple-mobile-web-app-title' },
	STATUS_BAR_STYLE: { content: 'black-translucent', name: 'apple-mobile-web-app-status-bar-style' }
} as const;

// Splash screen HTML link tag prefix used by generateSplashScreenLinks
export const SPLASH_LINK = {
	DARK_MEDIA_SUFFIX: ' and (prefers-color-scheme: dark)',
	HTML: '<link rel="apple-touch-startup-image"'
} as const;
