import { build } from 'esbuild';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';
import type { Plugin } from 'vite';

const __dirname = dirname(fileURLToPath(import.meta.url));

const VENDORS_DIR = resolve(__dirname, '../src/lib/vendors');
const VIRTUAL_ID = 'virtual:nerdamer';
const RESOLVED_ID = '\0' + VIRTUAL_ID;

/**
 * Bundle the vendored nerdamer-prime source into a minified IIFE string,
 * exposed as the `virtual:nerdamer` module. Flags mirror the upstream
 * build (esbuild --bundle --minify --format=iife --global-name=nerdamer),
 * so only human readable source lives in the repo and minification is a
 * build artifact. Vendored under src/lib/vendors/, upstream snapshot:
 * https://github.com/together-science/nerdamer-prime/commit/1936145f8af306ec0d883b9bfd7730aedd175c24
 */
export function nerdamerPlugin(): Plugin {
	let bundled: string | null = null;

	return {
		name: 'llamacpp:nerdamer',
		resolveId(id) {
			return id === VIRTUAL_ID ? RESOLVED_ID : undefined;
		},
		async load(id) {
			if (id !== RESOLVED_ID) return undefined;
			if (bundled === null) {
				const result = await build({
					entryPoints: [resolve(VENDORS_DIR, 'nerdamer-prime/all.js')],
					bundle: true,
					minify: true,
					format: 'iife',
					globalName: 'nerdamer',
					alias: {
						'big-integer': resolve(VENDORS_DIR, 'big-integer/BigInteger.js'),
						'decimal.js': resolve(VENDORS_DIR, 'decimal.js/decimal.js')
					},
					write: false,
					logLevel: 'silent'
				});
				bundled = result.outputFiles[0].text;
			}
			return `export default ${JSON.stringify(bundled)};`;
		}
	};
}
