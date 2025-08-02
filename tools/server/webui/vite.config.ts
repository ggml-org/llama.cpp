import devtoolsJson from 'vite-plugin-devtools-json';
import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { gzipSync } from 'zlib';
import { resolve } from 'path';

// Custom plugin to compress index.html with gzip after build
function gzipCompressionPlugin() {
	return {
		name: 'gzip-compression',
		apply: 'build' as const,
		closeBundle() {
			// Use setTimeout to ensure SvelteKit adapter has finished writing files
			setTimeout(() => {
				try {
					const indexPath = resolve('../public/index.html');
					const gzipPath = resolve('../public/index.html.gz');

					// Check if index.html exists
					if (!existsSync(indexPath)) {
						console.warn('⚠️  index.html not found, skipping gzip compression');
						return;
					}

					// Read the index.html file
					const indexContent = readFileSync(indexPath);

					// Compress with gzip
					const compressed = gzipSync(indexContent);

					// Write the compressed file
					writeFileSync(gzipPath, compressed);

					console.log('✓ Created index.html.gz');
				} catch (error) {
					console.error('Failed to create gzip file:', error);
				}
			}, 100); // Small delay to ensure SvelteKit adapter finishes
		}
	};
}

export default defineConfig({
	plugins: [tailwindcss(), sveltekit(), devtoolsJson(), gzipCompressionPlugin()],
	test: {
		projects: [
			{
				extends: './vite.config.ts',
				test: {
					name: 'client',
					environment: 'browser',
					browser: {
						enabled: true,
						provider: 'playwright',
						instances: [{ browser: 'chromium' }]
					},
					include: ['src/**/*.svelte.{test,spec}.{js,ts}'],
					exclude: ['src/lib/server/**'],
					setupFiles: ['./vitest-setup-client.ts']
				}
			},
			{
				extends: './vite.config.ts',
				test: {
					name: 'server',
					environment: 'node',
					include: ['src/**/*.{test,spec}.{js,ts}'],
					exclude: ['src/**/*.svelte.{test,spec}.{js,ts}']
				}
			}
		]
	}
});
