import { BRAND_FAVICONS, writeBrandFavicons } from '../../scripts/brand-favicons';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

const require = createRequire(import.meta.url);

describe('writeBrandFavicons', () => {
	let tmpDir: string;

	beforeEach(() => {
		tmpDir = mkdtempSync(join(tmpdir(), 'favicon-'));
	});

	afterEach(() => {
		rmSync(tmpDir, { force: true, recursive: true });
	});

	it("writes llama-brand's light and dark favicons unchanged", () => {
		writeBrandFavicons(tmpDir);

		for (const [fileName, specifier] of Object.entries(BRAND_FAVICONS)) {
			expect(readFileSync(join(tmpDir, fileName), 'utf-8')).toBe(
				readFileSync(require.resolve(specifier), 'utf-8')
			);
		}
	});

	it('draws the light favicon in ink and the dark one in light', () => {
		writeBrandFavicons(tmpDir);

		expect(readFileSync(join(tmpDir, 'favicon.svg'), 'utf-8')).toContain('fill="#111111"');
		expect(readFileSync(join(tmpDir, 'favicon-dark.svg'), 'utf-8')).toContain('fill="#FAFAFA"');
	});
});
