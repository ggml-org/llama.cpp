/**
 * Build utility functions for llama.cpp UI post-build processing.
 * These are called by the Vite plugin (vite-plugin-llama-cpp-build.ts)
 * and are exported for potential reuse in CI/CD scripts.
 */

import { readdirSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
	REGEX_PATTERNS,
	APPLE_DEVICES,
	BUNDLE_PATHS,
	SVELTEKIT_NORMALIZED,
	SPLASH_LINK
} from '../src/lib/constants/pwa';
import { NEWLINE } from '../src/lib/constants/code';
import type { SplashDimensions } from '../src/lib/types';
import { SplashOrientation } from '../src/lib/enums/splash.enums';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// SW-cached bundle URL: ?cache=true enables the SW to precache the app for PWA install.
const BUNDLE_CACHED = (path: string) => `${path}?cache=true`;

/** Resolve explicit build version from env var. Returns null if not set. */
export function getExplicitVersion(): string | null {
	const explicit = process.env.LLAMA_UI_VERSION;
	return explicit?.trim() || null;
}

/** Resolve build version: explicit env var (from CMake) > git hash > fallback. */
export function resolveBuildVersion(): string {
	const explicit = getExplicitVersion();
	if (explicit) {
		return explicit;
	}

	try {
		const gitHash = execSync('git rev-parse --short HEAD', {
			cwd: resolve(__dirname, '..'),
			encoding: 'utf-8'
		}).trim();
		const epoch = Math.floor(Date.now() / 1000);
		return `${gitHash}-${epoch}`;
	} catch {
		return `fallback-${Date.now()}`;
	}
}

/**
 * Generate iOS splash screen <link> tags from generated apple-splash-*.png files.
 * Returns an array of HTML link strings to be injected into the page head.
 */
export function generateSplashScreenLinks(outDir: string): string[] {
	const files = readdirSync(outDir).filter((f) => f.match(REGEX_PATTERNS.SPLASH_FILE));
	if (files.length === 0) return [];

	const dimMap = new Map<string, SplashDimensions>();
	for (const [dims, spec] of Object.entries(APPLE_DEVICES)) {
		const [w, h] = dims.split('x').map(Number);
		// logical-point dimensions
		dimMap.set(`${w}x${h}`, { deviceW: spec.width, deviceH: spec.height, dpr: spec.dpr });
		dimMap.set(`${h}x${w}`, { deviceW: spec.width, deviceH: spec.height, dpr: spec.dpr });
		// pixel dimensions (used by actual generated splash files)
		dimMap.set(`${w * spec.dpr}x${h * spec.dpr}`, {
			deviceW: spec.width,
			deviceH: spec.height,
			dpr: spec.dpr
		});
		dimMap.set(`${h * spec.dpr}x${w * spec.dpr}`, {
			deviceW: spec.width,
			deviceH: spec.height,
			dpr: spec.dpr
		});
	}

	const lightLinks: string[] = [];
	const darkLinks: string[] = [];

	for (const file of files) {
		const match = file.match(REGEX_PATTERNS.SPLASH_FILE);
		if (!match) continue;
		const orientation = match[1] as SplashOrientation;
		const isDark = !!match[2];
		const pixelW = parseInt(match[3]);
		const pixelH = parseInt(match[4]);

		const key = `${pixelW}x${pixelH}`;
		const spec = dimMap.get(key);
		if (!spec) {
			console.warn(`⚠ Unknown splash screen dimensions: ${key} (${file})`);
			continue;
		}

		const { deviceW, deviceH, dpr } = spec;
		const media = `screen and (device-width: ${deviceW}px) and (device-height: ${deviceH}px) and (-webkit-device-pixel-ratio: ${dpr}) and (orientation: ${orientation})`;
		const href = `./${file}`;

		if (isDark) {
			darkLinks.push(
				`${SPLASH_LINK.HTML} media="${media}${SPLASH_LINK.DARK_MEDIA_SUFFIX}" href="${href}">`
			);
		} else {
			lightLinks.push(`${SPLASH_LINK.HTML} media="${media}" href="${href}">`);
		}
	}

	return [...lightLinks, ...darkLinks];
}

/** Rewrite bundle paths in content: hashed paths → static names with ?cache=true.
 * This allows the SW to precache the PWA entry while keeping the uncached URL fresh. */
export function rewriteBundlePaths(_content: string): string {
	let result = _content;
	result = result.replace(REGEX_PATTERNS.BUNDLE_JS, BUNDLE_CACHED(BUNDLE_PATHS.JS));
	result = result.replace(REGEX_PATTERNS.BUNDLE_CSS, BUNDLE_CACHED(BUNDLE_PATHS.CSS));
	result = result.replace(REGEX_PATTERNS.SVELTEKIT_HASH, SVELTEKIT_NORMALIZED);
	return result;
}

/**
 * Fix sw.js: rewrite _app paths, favicon.svg → favicon.ico, workbox-*.js → workbox.js.
 * Use ?cache=true for all SW-cached assets so the PWA is installable.
 * The uncached entry (/ without param) stays fresh and triggers version detection.
 * Inject build version as a comment so SW content differs between builds,
 * triggering browser to detect a new service worker and fire needRefresh.
 */
export function fixServiceWorkerContent(content: string, buildVersion: string): string {
	let swContent = content;
	swContent = swContent.replace(REGEX_PATTERNS.BUNDLE_JS, BUNDLE_CACHED(BUNDLE_PATHS.JS));
	swContent = swContent.replace(REGEX_PATTERNS.BUNDLE_CSS, BUNDLE_CACHED(BUNDLE_PATHS.CSS));
	swContent = swContent.replace(REGEX_PATTERNS.FAVICON_SVG, '"favicon.ico"');
	swContent = swContent.replace(REGEX_PATTERNS.VERSION_JSON_APP, '"version.json"');
	swContent = swContent.replace(REGEX_PATTERNS.WORKBOX_IMPORT, '"./workbox"');
	swContent = swContent.replace(
		REGEX_PATTERNS.PRECACHE_BUNDLE_JS,
		`"${BUNDLE_CACHED(BUNDLE_PATHS.JS)}"`
	);
	swContent = swContent.replace(
		REGEX_PATTERNS.PRECACHE_BUNDLE_CSS,
		`"${BUNDLE_CACHED(BUNDLE_PATHS.CSS)}"`
	);

	// Navigation precache entry and fallback: SW caches the ?cache=true variant
	// for the PWA entry, while the uncached / stays fresh for version detection.
	swContent = swContent.replace('url:"/",revision', 'url:"./?cache=true",revision');
	swContent = swContent.replace('url:"./",revision', 'url:"./?cache=true",revision');
	swContent = swContent.replace(
		'createHandlerBoundToURL("/")',
		'createHandlerBoundToURL("./?cache=true")'
	);
	swContent = swContent.replace(
		'createHandlerBoundToURL("./")',
		'createHandlerBoundToURL("./?cache=true")'
	);

	return '// Build: ' + buildVersion + NEWLINE + swContent;
}
