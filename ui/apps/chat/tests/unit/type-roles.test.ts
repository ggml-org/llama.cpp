import { readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

// Type is set through the roles in ui/lib/ui/src/lib/styles/tokens/semantic.css (text-body-sm, text-label,
// text-caption ...), never Tailwind's own sizes or a size in brackets: see Typography in DESIGN.md.
const ui = fileURLToPath(new URL('../../../../', import.meta.url));
const ROOTS = ['apps/chat/src', 'apps/chat/tests/stories', 'lib/ui/src', 'lib/ui/tests/stories'];
const SOURCE = /\.(svelte|ts|js|css|svx)$/;
const TAILWIND_SIZE = /(?<![\w-])text-(?:xs|sm|base|lg|xl|[2-9]xl)(?![\w-])/g;
// a size relative to the text around it (text-[0.875em]) is not one off the scale
const BRACKET_SIZE = /(?<![\w-])text-\[[\d.]+(?:px|rem)\]/g;

function sources(dir: string): string[] {
	return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
		const path = join(dir, entry.name);

		if (entry.isDirectory()) return sources(path);

		return SOURCE.test(entry.name) ? [path] : [];
	});
}

function matches(pattern: RegExp): string[] {
	return ROOTS.flatMap((root) => sources(join(ui, root))).flatMap((path) =>
		[...readFileSync(path, 'utf8').matchAll(pattern)].map(
			([match]) => `${relative(ui, path)}: ${match}`
		)
	);
}

describe('type roles', () => {
	it('uses no Tailwind text size', () => {
		expect(matches(TAILWIND_SIZE)).toEqual([]);
	});

	it('uses no text size in brackets', () => {
		expect(matches(BRACKET_SIZE)).toEqual([]);
	});
});
