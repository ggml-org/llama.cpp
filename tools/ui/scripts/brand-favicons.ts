import { FAVICON_PATHS } from '../src/lib/constants/pwa.constants';
import { copyFileSync, mkdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const DEFAULT_OUT_DIR = resolve(HERE, '..', 'static');
const require = createRequire(import.meta.url);

/** The web favicons llama-brand draws for the UI, optically centred, keyed by the file they become. */
export const BRAND_FAVICONS = {
	[FAVICON_PATHS.SVG_DARK]: 'llama-brand/logo/svg/app/web/favicon-dark.svg',
	[FAVICON_PATHS.SVG_LIGHT]: 'llama-brand/logo/svg/app/web/favicon.svg'
} as const;

/**
 * Copy llama-brand's light and dark favicons into the static directory, where
 * they are served as they are and feed the PWA asset generator.
 */
export function writeBrandFavicons(outDir: string = DEFAULT_OUT_DIR): void {
	mkdirSync(outDir, { recursive: true });

	for (const [fileName, specifier] of Object.entries(BRAND_FAVICONS)) {
		copyFileSync(require.resolve(specifier), resolve(outDir, fileName));
	}
}
