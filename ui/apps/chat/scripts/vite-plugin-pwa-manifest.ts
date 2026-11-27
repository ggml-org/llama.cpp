import { HEADERS } from '../src/lib/constants/headers.constants.ts';
import {
	PWA_FILE_PATHS,
	PWA_MANIFEST_PLUGIN,
	pwaManifest
} from '../src/lib/constants/pwa.constants.ts';
import type { Plugin } from 'vite';

// the build writes files relative to its output directory: the path without its leading slash
const MANIFEST_FILE = PWA_FILE_PATHS.MANIFEST.slice(1);

/**
 * Serve manifest.webmanifest in dev and emit it next to the client build output.
 * The adapter copies the client output, so the file lands at the root of the build.
 */
export function pwaManifestPlugin(): Plugin {
	let manifestSource = '';

	return {
		// named from the env Vite loaded (.env files too): this module runs in Node, where
		// import.meta.env is not set and APP_NAME would fall back to its default
		configResolved(config) {
			manifestSource = JSON.stringify(pwaManifest(config.env));
		},
		configureServer(server) {
			server.middlewares.use(PWA_FILE_PATHS.MANIFEST, (_req, res) => {
				res.setHeader(HEADERS.CONTENT_TYPE, PWA_MANIFEST_PLUGIN.CONTENT_TYPE);
				res.end(manifestSource);
			});
		},
		generateBundle() {
			if (this.environment.name !== PWA_MANIFEST_PLUGIN.ENVIRONMENT) return;

			this.emitFile({ fileName: MANIFEST_FILE, source: manifestSource, type: 'asset' });
		},
		name: PWA_MANIFEST_PLUGIN.NAME
	};
}
