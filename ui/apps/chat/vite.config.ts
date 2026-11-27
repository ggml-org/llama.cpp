import { buildInfoPlugin } from './scripts/vite-plugin-build-info.ts';
import { llamaUiPlugin } from './scripts/vite-plugin-llama-ui.ts';
import { nerdamerPlugin } from './scripts/vite-plugin-nerdamer.ts';
import { pwaManifestPlugin } from './scripts/vite-plugin-pwa-manifest.ts';
import { relativizeBasePlugin } from './scripts/vite-plugin-relativize-base.ts';
import { splashScreenPlugin } from './scripts/vite-plugin-splash-screen.ts';
import { storybookTest } from '@storybook/addon-vitest/vitest-plugin';
import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import tailwindcss from '@tailwindcss/vite';
import { playwright } from '@vitest/browser-playwright';
import { mdsvex } from 'mdsvex';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';
import { defineConfig, loadEnv, searchForWorkspaceRoot } from 'vite';

const __dirname = dirname(fileURLToPath(import.meta.url));
const LLAMA_VOICE_SOURCE = resolve(__dirname, '../../lib/voice/src');
// Llama Hugging Face has no build step in the repo either: the app compiles it from source
const LLAMA_HUGGINGFACE_ROOT = resolve(__dirname, '../../lib/huggingface');
// the harness has no build step in the repo: the app compiles it from source
const LLAMA_HARNESS_ENTRY = resolve(__dirname, '../../lib/harness/src/index.ts');
const LLAMA_HARNESS_TRACE_ENTRY = resolve(__dirname, '../../lib/harness/src/trace/index.ts');
// what a chat-only build (VITE_PUBLIC_AGENTIC=false) gets in place of the harness: its trace phrasing, no loop
const LLAMA_HARNESS_CHAT_ONLY_ENTRY = resolve(
	__dirname,
	'src/lib/stores/agentic/harness-chat-only.ts'
);
// what a build without sharing to Hugging Face (a build that is not asked for it with
// VITE_PUBLIC_HUGGING_FACE=true) gets in place of the package's Svelte entry: an account that is
// never signed in, and components that draw nothing
const LLAMA_HUGGINGFACE_DISABLED_ENTRY = resolve(
	__dirname,
	'src/lib/stores/huggingface-disabled.ts'
);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const browserBaseConfig: any = {
	enabled: true,
	instances: [{ browser: 'chromium' }],
	// CI runs the bundled Chromium without its sandbox, which the runners can't start;
	// locally, the installed (auto-updated) Chrome with its sandbox on, see playwright.config.ts
	provider: playwright({
		launchOptions: process.env.CI
			? { args: ['--no-sandbox'] }
			: { channel: 'chrome', chromiumSandbox: true }
	})
};
// CMake sets LLAMA_UI_OUT_DIR to the staging dir under the build tree; manual
// `npm run build` runs without the env var default to ./dist.
const outDir = process.env.LLAMA_UI_OUT_DIR ?? './dist';

export default defineConfig(({ command, mode }) => {
	const env = loadEnv(mode, process.cwd(), 'VITE_PUBLIC_');
	const SERVER_ORIGIN = env.VITE_PUBLIC_SERVER_ORIGIN || 'http://localhost:9931';
	// A build has sharing to Hugging Face only when asked for: VITE_PUBLIC_HUGGING_FACE=true. The
	// dev server, the tests and Storybook have it unless told otherwise (=false).
	const huggingFace = env.VITE_PUBLIC_HUGGING_FACE
		? env.VITE_PUBLIC_HUGGING_FACE === 'true'
		: command === 'serve';

	return {
		build: {
			assetsInlineLimit: 32000,
			chunkSizeWarningLimit: 3072,
			minify: true
		},

		// what the app reads as HUGGING_FACE_ENABLED, so it agrees with the alias below
		define: {
			'import.meta.env.VITE_PUBLIC_HUGGING_FACE': JSON.stringify(String(huggingFace))
		},

		plugins: [
			llamaUiPlugin(),
			tailwindcss(),
			sveltekit({
				adapter: adapter({
					assets: outDir,
					fallback: 'index.html',
					pages: outDir,
					precompress: false,
					strict: true
				}),
				extensions: ['.svelte', '.svx'],
				output: { bundleStrategy: 'split' },
				paths: { relative: true },
				// Consult https://svelte.dev/docs/kit/integrations
				// for more information about preprocessors
				// mdsvex is for content pages we write ourselves (e.g. release notes reachable from the
				// chat app): .svx files compiled to components at build time. No page uses it yet.
				// It can't render chat input or messages, which are typed at runtime and go through
				// the remark/rehype pipeline in MarkdownContent instead.
				preprocess: [vitePreprocess(), mdsvex()],
				router: { type: 'hash' },
				// registered by usePwa() so the update prompt can drive it
				serviceWorker: { register: false }
			}),
			pwaManifestPlugin(),
			splashScreenPlugin(),
			buildInfoPlugin(),
			nerdamerPlugin(),
			relativizeBasePlugin()
		],

		resolve: {
			alias: [
				// Llama Voice has no build step in the repo: the app compiles it from source. Exact
				// matches, so the package's root does not swallow its ./svelte entry
				{
					find: /^@ggml-org\/llama-voice\/svelte$/,
					replacement: resolve(LLAMA_VOICE_SOURCE, 'svelte/index.ts')
				},
				{ find: /^@ggml-org\/llama-voice$/, replacement: resolve(LLAMA_VOICE_SOURCE, 'index.ts') },
				{
					find: /^@ggml-org\/llama-huggingface\/svelte$/,
					replacement: huggingFace
						? resolve(LLAMA_HUGGINGFACE_ROOT, 'src/svelte/index.ts')
						: LLAMA_HUGGINGFACE_DISABLED_ENTRY
				},
				{
					find: /^@ggml-org\/llama-huggingface$/,
					replacement: resolve(LLAMA_HUGGINGFACE_ROOT, 'src/index.ts')
				},
				{ find: /^@ggml-org\/llama-harness\/trace$/, replacement: LLAMA_HARNESS_TRACE_ENTRY },
				{
					find: /^@ggml-org\/llama-harness$/,
					replacement:
						env.VITE_PUBLIC_AGENTIC === 'false'
							? LLAMA_HARNESS_CHAT_ONLY_ENTRY
							: LLAMA_HARNESS_ENTRY
				},
				{ find: 'katex-fonts', replacement: resolve('node_modules/katex/dist/fonts') }
			]
		},

		server: {
			fs: {
				allow: [
					searchForWorkspaceRoot(process.cwd()),
					resolve(__dirname, 'tests'),
					LLAMA_VOICE_SOURCE,
					LLAMA_HUGGINGFACE_ROOT,
					resolve(__dirname, '../../lib/harness')
				]
			},
			headers: {
				'Cross-Origin-Embedder-Policy': 'require-corp',
				'Cross-Origin-Opener-Policy': 'same-origin'
			},
			proxy: {
				'/cors-proxy': SERVER_ORIGIN,
				'/models': SERVER_ORIGIN,
				'/props': SERVER_ORIGIN,
				'/slots': SERVER_ORIGIN,
				'/tools': SERVER_ORIGIN,
				// ws: realtime transcription (/v1/realtime) is a WebSocket
				'/v1': { target: SERVER_ORIGIN, ws: true }
			}
		},

		test: {
			projects: [
				{
					extends: './vite.config.ts',
					test: {
						browser: browserBaseConfig,
						include: ['tests/client/**/*.svelte.{test,spec}.{js,ts}'],
						name: 'client',
						setupFiles: ['./vitest-setup-client.ts']
					}
				},

				{
					extends: './vite.config.ts',
					test: {
						environment: 'node',
						include: ['tests/unit/**/*.{test,spec}.{js,ts}'],
						name: 'unit'
					}
				},

				{
					extends: './vite.config.ts',
					plugins: [
						storybookTest({
							storybookScript: 'pnpm run storybook --no-open'
						})
					],
					test: {
						browser: { ...browserBaseConfig, instances: [{ browser: 'chromium', headless: true }] },
						// Llama UI's stories, shown in this Storybook too, are tested by the library itself (ui/lib/ui)
						exclude: ['../../lib/ui/tests/**'],
						name: 'ui'
					}
				}
			]
		}
	};
});
