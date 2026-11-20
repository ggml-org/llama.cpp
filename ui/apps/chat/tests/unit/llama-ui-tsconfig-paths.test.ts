import { llamaUiPaths } from '../../scripts/llama-ui.js';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

// tsconfig.json spells out Llama UI's paths for TypeScript; scripts/llama-ui.js derives the same
// from the library's package.json, so a dependency the library gains must reach tsconfig.json too
const APP_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '../..');

function absolute(paths: Record<string, string[]>): Record<string, string[]> {
	return Object.fromEntries(
		Object.entries(paths).map(([specifier, targets]) => [
			specifier,
			targets.map((target) => resolve(APP_DIR, target))
		])
	);
}

describe("tsconfig.json's Llama UI paths", () => {
	it('match what scripts/llama-ui.js derives from the library', () => {
		const tsconfig = JSON.parse(readFileSync(resolve(APP_DIR, 'tsconfig.json'), 'utf8'));
		const expected = absolute(llamaUiPaths(resolve(APP_DIR, 'node_modules')));
		const actual = absolute(tsconfig.compilerOptions.paths);

		for (const [specifier, targets] of Object.entries(expected)) {
			expect(actual[specifier], specifier).toEqual(targets);
		}
	});
});
