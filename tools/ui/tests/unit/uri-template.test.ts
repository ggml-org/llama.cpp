import { describe, it, expect } from 'vitest';
import {
	extractTemplateVariables,
	expandTemplate,
	isTemplateComplete,
	normalizeResourceUri
} from '../../src/lib/utils/uri-template';
import { URI_TEMPLATE_OPERATORS } from '../../src/lib/constants/uri-template';

describe('extractTemplateVariables', () => {
	it('extracts simple variables', () => {
		const vars = extractTemplateVariables('file:///{path}');
		expect(vars).toEqual([{ name: 'path', operator: '' }]);
	});

	it('extracts multiple variables', () => {
		const vars = extractTemplateVariables('db://{schema}/{table}');
		expect(vars).toEqual([
			{ name: 'schema', operator: '' },
			{ name: 'table', operator: '' }
		]);
	});

	it('extracts variables with operators', () => {
		const vars = extractTemplateVariables('http://example.com{+path}');
		expect(vars).toEqual([{ name: 'path', operator: URI_TEMPLATE_OPERATORS.RESERVED }]);
	});

	it('extracts comma-separated variable lists', () => {
		const vars = extractTemplateVariables('{x,y,z}');
		expect(vars).toEqual([
			{ name: 'x', operator: '' },
			{ name: 'y', operator: '' },
			{ name: 'z', operator: '' }
		]);
	});

	it('deduplicates variable names', () => {
		const vars = extractTemplateVariables('{name}/{name}');
		expect(vars).toEqual([{ name: 'name', operator: '' }]);
	});

	it('handles fragment expansion', () => {
		const vars = extractTemplateVariables('http://example.com/page{#section}');
		expect(vars).toEqual([{ name: 'section', operator: URI_TEMPLATE_OPERATORS.FRAGMENT }]);
	});

	it('handles path segment expansion', () => {
		const vars = extractTemplateVariables('http://example.com{/path}');
		expect(vars).toEqual([{ name: 'path', operator: URI_TEMPLATE_OPERATORS.PATH_SEGMENT }]);
	});

	it('returns empty array for template without variables', () => {
		const vars = extractTemplateVariables('http://example.com/static');
		expect(vars).toEqual([]);
	});

	it('strips explode modifier', () => {
		const vars = extractTemplateVariables('{list*}');
		expect(vars).toEqual([{ name: 'list', operator: '' }]);
	});

	it('strips prefix modifier', () => {
		const vars = extractTemplateVariables('{value:5}');
		expect(vars).toEqual([{ name: 'value', operator: '' }]);
	});
});

describe('expandTemplate', () => {
	it('expands simple variable', () => {
		const result = expandTemplate('file:///{path}', { path: 'src/main.rs' });
		expect(result).toBe('file:///src%2Fmain.rs');
	});

	it('expands reserved variable (no encoding)', () => {
		const result = expandTemplate('file:///{+path}', { path: 'src/main.rs' });
		expect(result).toBe('file:///src/main.rs');
	});

	it('expands multiple variables', () => {
		const result = expandTemplate('db://{schema}/{table}', {
			schema: 'public',
			table: 'users'
		});
		expect(result).toBe('db://public/users');
	});

	it('leaves empty for missing variables', () => {
		const result = expandTemplate('{missing}', {});
		expect(result).toBe('');
	});

	it('expands fragment', () => {
		const result = expandTemplate('http://example.com/page{#section}', {
			section: 'intro'
		});
		expect(result).toBe('http://example.com/page#intro');
	});

	it('expands path segments', () => {
		const result = expandTemplate('http://example.com{/path}', { path: 'docs' });
		expect(result).toBe('http://example.com/docs');
	});

	it('expands query parameters', () => {
		const result = expandTemplate('http://example.com{?q}', { q: 'search term' });
		expect(result).toBe('http://example.com?q=search%20term');
	});

	it('expands multiple query parameters', () => {
		const result = expandTemplate('http://example.com{?q,sort}', {
			q: 'search term',
			sort: 'descending'
		});
		expect(result).toBe('http://example.com?q=search%20term&sort=descending');
	});

	it('keeps static parts unchanged', () => {
		const result = expandTemplate('http://example.com/static', {});
		expect(result).toBe('http://example.com/static');
	});
});

describe('isTemplateComplete', () => {
	it('returns true when all variables are filled', () => {
		expect(isTemplateComplete('file:///{path}', { path: 'test.txt' })).toBe(true);
	});

	it('returns false when a variable is missing', () => {
		expect(isTemplateComplete('db://{schema}/{table}', { schema: 'public' })).toBe(false);
	});

	it('returns false when a variable is empty', () => {
		expect(isTemplateComplete('file:///{path}', { path: '' })).toBe(false);
	});

	it('returns false when a variable is whitespace only', () => {
		expect(isTemplateComplete('file:///{path}', { path: '   ' })).toBe(false);
	});

	it('returns true for template without variables', () => {
		expect(isTemplateComplete('http://example.com/static', {})).toBe(true);
	});

	it('returns true when all multiple variables are filled', () => {
		expect(isTemplateComplete('db://{schema}/{table}', { schema: 'public', table: 'users' })).toBe(
			true
		);
	});
});

describe('normalizeResourceUri', () => {
	it('passes through a normal URI unchanged', () => {
		expect(normalizeResourceUri('svelte://svelte/$effect.md')).toBe('svelte://svelte/$effect.md');
	});

	it('normalizes triple-slash URIs from path-style template expansion', () => {
		expect(normalizeResourceUri('svelte:///svelte/$effect.md')).toBe('svelte://svelte/$effect.md');
	});

	it('normalizes quadruple-slash URIs', () => {
		expect(normalizeResourceUri('svelte:////svelte/$effect.md')).toBe('svelte://svelte/$effect.md');
	});

	it('handles file:// URIs', () => {
		expect(normalizeResourceUri('file:///home/user/doc.txt')).toBe('file://home/user/doc.txt');
	});

	it('handles http URIs unchanged', () => {
		expect(normalizeResourceUri('http://example.com/path')).toBe('http://example.com/path');
	});

	it('returns non-URI strings unchanged', () => {
		expect(normalizeResourceUri('not-a-uri')).toBe('not-a-uri');
	});
});
