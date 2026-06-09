import { describe, expect, it } from 'vitest';
import { rewriteBundlePaths } from '../../scripts/build-utils';

describe('rewriteBundlePaths', () => {
	it('uses ?v=<version> when buildVersion is provided', () => {
		const input = `<script type="module" src="./_app/immutable/bundle.abc123.js"></script>
<link rel="stylesheet" href="./_app/immutable/assets/bundle.def456.css">`;
		const result = rewriteBundlePaths(input, '1.0.0');

		expect(result).toMatch(/\.\/bundle\.js\?v=1\.0\.0/);
		expect(result).toMatch(/\.\/bundle\.css\?v=1\.0\.0/);
	});

	it('uses Vite content hash when buildVersion is not provided', () => {
		const input = `<script type="module" src="./_app/immutable/bundle.abc123.js"></script>
<link rel="stylesheet" href="./_app/immutable/assets/bundle.def456.css">`;
		const result = rewriteBundlePaths(input);

		expect(result).toMatch(/\.\/bundle\.js\?([a-zA-Z0-9_-]+)/);
		expect(result).toMatch(/\.\/bundle\.css\?([a-zA-Z0-9_-]+)/);
	});

	it('replaces __sveltekit__<hash> in both cases', () => {
		const withVersion = rewriteBundlePaths('__sveltekit_abc123', '1.0.0');
		expect(withVersion).toBe('__sveltekit__');
		expect(withVersion).not.toMatch(/__sveltekit_[a-z0-9]+/);

		const withoutVersion = rewriteBundlePaths('__sveltekit_abc123');
		expect(withoutVersion).toBe('__sveltekit__');
		expect(withoutVersion).not.toMatch(/__sveltekit_[a-z0-9]+/);
	});
});
