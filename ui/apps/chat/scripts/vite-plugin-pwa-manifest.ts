import { HEADERS } from '../src/lib/constants/headers.constants';
import {
	PWA_FILE_PATHS,
	PWA_MANIFEST,
	PWA_MANIFEST_PLUGIN
} from '../src/lib/constants/pwa.constants';
import type { Plugin } from 'vite';

const MANIFEST_SOURCE = JSON.stringify(PWA_MANIFEST);
// the build writes files relative to its output directory: the path without its leading slash
const MANIFEST_FILE = PWA_FILE_PATHS.MANIFEST.slice(1);

/**
 * Serve manifest.webmanifest in dev and emit it next to the client build output.
 * The adapter copies the client output, so the file lands at the root of the build.
 */
export function pwaManifestPlugin(): Plugin {
	return {
		configureServer(server) {
			server.middlewares.use(PWA_FILE_PATHS.MANIFEST, (_req, res) => {
				res.setHeader(HEADERS.CONTENT_TYPE, PWA_MANIFEST_PLUGIN.CONTENT_TYPE);
				res.end(MANIFEST_SOURCE);
			});
		},
		generateBundle() {
			if (this.environment.name !== PWA_MANIFEST_PLUGIN.ENVIRONMENT) return;

			this.emitFile({ fileName: MANIFEST_FILE, source: MANIFEST_SOURCE, type: 'asset' });
		},
		name: PWA_MANIFEST_PLUGIN.NAME
	};
}
