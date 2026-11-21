import { LLAMA_UI_ALIASES, LLAMA_UI_DEPENDENCIES, LLAMA_UI_ROOT } from './llama-ui.js';
import { fileURLToPath } from 'node:url';
import { type Plugin, transformWithOxc } from 'vite';

const TYPESCRIPT_FILE = /\.ts$/;
// what the library's sources need of a tsconfig: imports and exports stay as written
const TSCONFIG = { compilerOptions: { verbatimModuleSyntax: true } };
// the app's tsconfig.json, for every module and for the dependency scan alike
const APP_TSCONFIG = fileURLToPath(new URL('../tsconfig.json', import.meta.url));
const LIBRARY_FILE = new RegExp(`^${LLAMA_UI_ROOT.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`);

/**
 * Compiles Llama UI (@ggml-org/llama-ui, in ui/lib/ui) from its source as part of this app, with
 * no install or build of the library itself (see ./llama-ui.js for the why):
 *
 * - its entry points resolve to their sources,
 * - what those sources import resolves from this app's node_modules,
 * - the dev server may serve them,
 * - their TypeScript is stripped here, and the app's tsconfig.json applies to them. Vite would
 *   read the tsconfig.json nearest to each file, and the library's extends one that only exists
 *   once the library is installed.
 */
export function llamaUiPlugin(): Plugin {
	return {
		config() {
			return {
				// the scan looks for each file's nearest tsconfig on its own, see `tsconfig` below
				optimizeDeps: { rolldownOptions: { tsconfig: APP_TSCONFIG } },
				oxc: { exclude: [/\.js$/, LIBRARY_FILE] },
				resolve: { alias: LLAMA_UI_ALIASES, dedupe: LLAMA_UI_DEPENDENCIES },
				server: { fs: { allow: [LLAMA_UI_ROOT] } },
				// one tsconfig for every module, instead of the nearest one, which for the library's
				// sources extends a config only its own install has
				tsconfig: APP_TSCONFIG
			};
		},
		enforce: 'pre',
		name: 'llama-ui',
		async transform(code, id) {
			const file = id.split('?')[0];

			if (!LIBRARY_FILE.test(file) || !TYPESCRIPT_FILE.test(file)) return;

			const result = await transformWithOxc(code, file, {
				lang: 'ts',
				sourcemap: true,
				tsconfig: TSCONFIG
			});

			return { code: result.code, map: result.map };
		}
	};
}
