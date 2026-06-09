import {
	readFileSync,
	writeFileSync,
	existsSync,
	readdirSync,
	copyFileSync,
	rmSync
} from 'node:fs';
import { resolve } from 'node:path';
import type { Plugin } from 'vite';
import { BUILD_CONFIG, REGEX_PATTERNS } from '../src/lib/constants/pwa';
import {
	resolveBuildVersion,
	generateSplashScreenLinks,
	rewriteBundlePaths,
	fixServiceWorkerContent
} from './build-utils';

let processed = false;

const OUTPUT_DIR = process.env.LLAMA_UI_OUT_DIR ?? BUILD_CONFIG.OUTPUT_DIR;

export function llamaCppBuildPlugin(): Plugin {
	return {
		name: 'llamacpp:build',
		apply: 'build',
		closeBundle() {
			setTimeout(() => {
				try {
					if (processed) return;
					processed = true;

					const buildVersion = resolveBuildVersion();
					const outDir = resolve(OUTPUT_DIR);
					const indexPath = resolve(outDir, 'index.html');
					if (!existsSync(indexPath)) return;

					let content = readFileSync(indexPath, 'utf-8');

					const splashLinks = generateSplashScreenLinks(outDir);
					if (splashLinks.length > 0) {
						console.log(`✓ Generated ${splashLinks.length} apple-splash link tags`);
						const splashHtml = splashLinks.map((l) => '\t\t' + l).join('\n');
						content = content.replace(/\t*<\/head>/, splashHtml + '\n\t\t</head>');
					}

					content = content.replace(/\r/g, '');
					content = BUILD_CONFIG.GUIDE_COMMENT + '\n' + content;
					content = rewriteBundlePaths(content, buildVersion);

					writeFileSync(indexPath, content, 'utf-8');
					console.log('✓ Updated index.html');

					const immutableDir = resolve(outDir, '_app/immutable');
					const bundleDir = resolve(outDir, '_app/immutable/assets');

					if (existsSync(immutableDir)) {
						const jsFiles = readdirSync(immutableDir).filter((f) =>
							f.match(REGEX_PATTERNS.BUNDLE_JS_FILE)
						);
						if (jsFiles.length > 0) {
							copyFileSync(resolve(immutableDir, jsFiles[0]), resolve(outDir, 'bundle.js'));
							const bundleJsPath = resolve(outDir, 'bundle.js');
							let bundleJs = readFileSync(bundleJsPath, 'utf-8');
							bundleJs = bundleJs.replace(REGEX_PATTERNS.SVELTEKIT_HASH, '__sveltekit__');
							writeFileSync(bundleJsPath, bundleJs, 'utf-8');
							console.log(`✓ Copied ${jsFiles[0]} -> bundle.js`);
						}
					}

					if (existsSync(bundleDir)) {
						const cssFiles = readdirSync(bundleDir).filter((f) =>
							f.match(REGEX_PATTERNS.BUNDLE_CSS_FILE)
						);
						if (cssFiles.length > 0) {
							copyFileSync(resolve(bundleDir, cssFiles[0]), resolve(outDir, 'bundle.css'));
							console.log(`✓ Copied ${cssFiles[0]} -> bundle.css`);
						}
					}

					const workboxFiles = readdirSync(outDir).filter((f) =>
						f.match(REGEX_PATTERNS.WORKBOX_FILE)
					);
					if (workboxFiles.length > 0) {
						copyFileSync(resolve(outDir, workboxFiles[0]), resolve(outDir, 'workbox.js'));
						rmSync(resolve(outDir, workboxFiles[0]), { force: true });
						console.log(`✓ Copied ${workboxFiles[0]} -> workbox.js`);
					}

					writeFileSync(resolve(outDir, 'version.json'), JSON.stringify({ version: buildVersion }));
					console.log('✓ Generated version.json:', buildVersion);

					const swPath = resolve(outDir, 'sw.js');
					if (existsSync(swPath)) {
						const swContent = readFileSync(swPath, 'utf-8');
						const fixedContent = fixServiceWorkerContent(swContent, buildVersion);
						writeFileSync(swPath, fixedContent, 'utf-8');
						console.log('✓ Fixed sw.js precache paths + injected build version');
					}

					const appDir = resolve(outDir, '_app');
					if (existsSync(appDir)) {
						rmSync(appDir, { recursive: true, force: true });
						console.log('✓ Removed _app directory');
					}
				} catch (error) {
					console.error('Failed to process build output:', error);
				}
			}, 100);
		}
	};
}
