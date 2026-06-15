import { writeFileSync, existsSync } from 'node:fs';
import { resolve } from 'path';
import type { Plugin } from 'vite';
import { BUILD_CONFIG } from '../src/lib/constants/pwa';

let processed = false;

const OUTPUT_DIR = process.env.LLAMA_UI_OUT_DIR ?? BUILD_CONFIG.OUTPUT_DIR;

/**
 * Write build.json with the llama.cpp release build number.
 *
 * LLAMA_BUILD_NUMBER is passed from CMake -> npm -> vite via env var.
 * Used for display of the current llama-server release (e.g. "b1234").
 */
export function buildInfoPlugin(): Plugin {
	return {
		name: 'llamacpp:build-info',
		apply: 'build',
		closeBundle() {
			setTimeout(() => {
				try {
					if (processed) return;
					processed = true;

					const buildNumber = process.env.LLAMA_BUILD_NUMBER || 'b0000';

					const outDir = resolve(OUTPUT_DIR);
					const indexPath = resolve(outDir, 'index.html');
					if (!existsSync(indexPath)) return;

					const buildJsonPath = resolve(outDir, 'build.json');
					writeFileSync(buildJsonPath, JSON.stringify({ version: buildNumber }), 'utf-8');
					console.log(`Created build.json (version: ${buildNumber})`);
				} catch (error) {
					console.error('Failed to write build.json:', error);
				}
			}, 100);
		}
	};
}
