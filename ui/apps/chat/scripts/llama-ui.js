import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

/**
 * Llama UI (@ggml-org/llama-ui, in ui/lib/ui) has no build step in the repo: this app compiles it
 * from source. vite.config.ts (through vite-plugin-llama-ui.ts) and tsconfig.json share what that
 * takes; tests/unit/llama-ui-tsconfig-paths.test.ts checks that tsconfig.json's paths match:
 *
 * - LLAMA_UI_EXPORTS maps the entry points of the package to their sources, as the exports of
 *   its package.json map them to its build, so an import reads the same either way.
 * - LLAMA_UI_DEPENDENCIES lists what those sources import. They sit outside this folder, so
 *   walking up from them would not find this app's node_modules, and would find a second copy
 *   of Svelte if ui/lib/ui has its own install. Both configs resolve these names from this
 *   app's node_modules instead.
 */
// with forward slashes on every platform, as Vite writes the paths of the modules it handles
const root = fileURLToPath(new URL('../../../lib/ui/', import.meta.url)).replace(/\\/g, '/');
const packageJson = JSON.parse(readFileSync(`${root}package.json`, 'utf8'));
const source = `${root}src/lib`;
/** [import specifier, source path]; `*` stands for one path segment or more. */
const LLAMA_UI_EXPORTS = [
	['@ggml-org/llama-ui/ui/*', `${source}/components/ui/*/index.ts`],
	['@ggml-org/llama-ui/app/*', `${source}/components/app/*/index.ts`],
	['@ggml-org/llama-ui/models', `${source}/components/models/index.ts`],
	['@ggml-org/llama-ui/constants', `${source}/constants/index.ts`],
	['@ggml-org/llama-ui/enums', `${source}/enums/index.ts`],
	['@ggml-org/llama-ui/code', `${source}/utils/index.ts`],
	['@ggml-org/llama-ui/utils', `${source}/components/ui/utils.ts`],
	['@ggml-org/llama-ui/voice', `${source}/components/voice/index.ts`],
	['@ggml-org/llama-ui/styles/*', `${source}/styles/*`],
	['@ggml-org/llama-ui/theme.css', `${source}/styles/theme.css`],
	['@ggml-org/llama-ui/tokens.css', `${source}/styles/tokens.css`],
	['@ggml-org/llama-ui/fonts.css', `${source}/styles/fonts.css`],
	['@ggml-org/llama-ui/surfaces.css', `${source}/styles/surfaces.css`],
	['@ggml-org/llama-ui', `${source}/index.ts`]
];

/** The folder of the library, ending in a slash. */
export const LLAMA_UI_ROOT = root;

/** The packages the library's sources import: its dependencies and peer dependencies. */
export const LLAMA_UI_DEPENDENCIES = Object.keys({
	...packageJson.dependencies,
	...packageJson.peerDependencies
}).sort();

/** The entry points as Vite aliases. */
export const LLAMA_UI_ALIASES = LLAMA_UI_EXPORTS.map(([specifier, path]) => ({
	find: new RegExp(`^${specifier.replace(/[.]/g, '\\.').replace('*', '(.+)')}$`),
	replacement: path.replace('*', '$1')
}));

/**
 * The entry points and the dependencies as TypeScript paths.
 * @param {string} nodeModules this app's node_modules, as the tsconfig that takes the paths sees it
 */
export function llamaUiPaths(nodeModules) {
	/** @type {Record<string, string[]>} */
	const paths = {};

	for (const [specifier, path] of LLAMA_UI_EXPORTS) paths[specifier] = [path];

	for (const name of LLAMA_UI_DEPENDENCIES) {
		paths[name] = [`${nodeModules}/${name}`];
		paths[`${name}/*`] = [`${nodeModules}/${name}/*`];
	}

	// Svelte declares the subpaths with no file of their own (svelte/reactivity, svelte/attachments) in
	// types/index.d.ts, which only its package exports point at; without this fallback, a library source
	// that imports one would get its types from a second copy of Svelte in ui/lib/ui/node_modules
	paths['svelte/*'] = [`${nodeModules}/svelte/*`, `${nodeModules}/svelte/types/index.d.ts`];

	// the icons sit behind lucide's package exports, which a path mapping does not read
	paths['@lucide/svelte/icons/*'] = [`${nodeModules}/@lucide/svelte/dist/icons/*`];

	return paths;
}
