import { describe, expect, it } from 'vitest';
import { rewriteBundlePaths } from '../../scripts/build-utils';

describe('rewriteBundlePaths', () => {
	it('uses ?cache=true for bundle.js and bundle.css', () => {
		const input = `<script type="module" src="./_app/immutable/bundle.abc123.js"></script>
<link rel="stylesheet" href="./_app/immutable/assets/bundle.def456.css">`;
		const result = rewriteBundlePaths(input);

		expect(result).toMatch(/\.\/bundle\.js\?cache=true/);
		expect(result).toMatch(/\.\/bundle\.css\?cache=true/);
	});

	it('replaces __sveltekit__<hash> with __sveltekit__', () => {
		const result = rewriteBundlePaths('__sveltekit_abc123');
		expect(result).toBe('__sveltekit__');
		expect(result).not.toMatch(/__sveltekit_[a-z0-9]+/);
	});
});
