import { describe, expect, it } from 'vitest';
import { parseFrontmatter, serializeFrontmatter } from '$lib/utils/frontmatter';
import {
	skillToMarkdown,
	markdownToSkill,
	composeSystemPromptWithAlwaysOnSkills,
	normalizeSkillName,
	validateSkillName,
	validateSkillDescription
} from '$lib/utils/skill-format';

describe('parseFrontmatter', () => {
	it('parses scalar frontmatter', () => {
		const input = `---
name: Hello
description: A greeting
---

# Heading

Body text`;
		const result = parseFrontmatter(input);
		expect(result.frontmatter.name).toBe('Hello');
		expect(result.frontmatter.description).toBe('A greeting');
		expect(result.body).toBe('# Heading\n\nBody text');
	});

	it('returns full input as body when no frontmatter', () => {
		const input = 'No frontmatter here';
		const result = parseFrontmatter(input);
		expect(result.frontmatter).toEqual({});
		expect(result.body).toBe('No frontmatter here');
	});

	it('treats unterminated frontmatter as body', () => {
		const input = `---
name: Never closes

# body`;
		const result = parseFrontmatter(input);
		expect(result.frontmatter).toEqual({});
		expect(result.body).toBe(input);
	});

	it('parses numeric timestamps as numbers', () => {
		const input = `---
name: T
last-modified: 1700000000000
---

body`;
		const result = parseFrontmatter(input);
		expect(result.frontmatter['last-modified']).toBe(1700000000000);
		expect(typeof result.frontmatter['last-modified']).toBe('number');
		expect(result.body).toBe('body');
	});

	it('parses booleans', () => {
		const input = `---
name: T
disable-model-invocation: true
---

body`;
		const result = parseFrontmatter(input);
		expect(result.frontmatter['disable-model-invocation']).toBe(true);
	});

	it('keeps string scalars quoted', () => {
		const input = `---
name: "Plain: text"
---

body`;
		const result = parseFrontmatter(input);
		expect(result.frontmatter.name).toBe('Plain: text');
	});
});

describe('serializeFrontmatter', () => {
	it('round-trips scalars without quoting when safe', () => {
		const md = serializeFrontmatter(
			{ name: 'Plain', description: 'A short description' },
			'Hello world'
		);
		expect(md).toContain('name: Plain');
		expect(md).toContain('description: A short description');
		expect(md.endsWith('Hello world')).toBe(true);
	});

	it('quotes values with YAML-special characters', () => {
		const md = serializeFrontmatter({ name: 'Title: with: colons', description: 'true' }, 'x');
		expect(md).toContain('name: "Title: with: colons"');
		expect(md).toContain('description: "true"');
	});

	it('serializes booleans', () => {
		const md = serializeFrontmatter({ name: 'T', 'disable-model-invocation': true }, 'b');
		expect(md).toContain('disable-model-invocation: true');
	});

	it('serializes numbers', () => {
		const md = serializeFrontmatter({ name: 'T', 'last-modified': 1700000000000 }, 'b');
		expect(md).toContain('last-modified: 1700000000000');
	});

	it('skips undefined fields', () => {
		const md = serializeFrontmatter({ name: 'T', description: undefined }, 'b');
		expect(md).not.toContain('description:');
	});
});

describe('skillToMarkdown / markdownToSkill round-trip', () => {
	it('preserves all fields', () => {
		const skill = {
			id: '11111111-1111-1111-1111-111111111111',
			name: 'code-reviewer',
			description: 'Reviews code for quality and security.',
			content: 'Review this code carefully.',
			lastModified: 1700000000000
		};

		const md = skillToMarkdown(skill);
		const parsed = markdownToSkill(md, skill.id);

		expect(parsed.name).toBe(skill.name);
		expect(parsed.description).toBe(skill.description);
		expect(parsed.content).toBe(skill.content);
		expect(parsed.lastModified).toBe(skill.lastModified);
		expect(parsed.id).toBe(skill.id);
	});

	it('drops unknown optional fields gracefully', () => {
		const md = skillToMarkdown({
			id: 'abc',
			name: 'x',
			description: 'A',
			content: 'Y',
			lastModified: 1
		});
		const parsed = markdownToSkill(md, 'abc');
		expect(parsed.name).toBe('x');
		expect(parsed.content).toBe('Y');
		expect(parsed.lastModified).toBe(1);
	});

	it('skips block without name', () => {
		const md = `---
description: missing name
---

body`;
		const parsed = markdownToSkill(md, undefined);
		expect(parsed.name).toBe('');
		expect(parsed.description).toBe('missing name');
	});

	it('preserves description when supplied', () => {
		const md = skillToMarkdown({
			name: 'X',
			description: 'A subtle one',
			content: 'body',
			lastModified: 1
		});
		const parsed = markdownToSkill(md, undefined);
		expect(parsed.description).toBe('A subtle one');
	});
});

describe('normalizeSkillName', () => {
	it('converts to lowercase', () => {
		expect(normalizeSkillName('Code Review')).toBe('code-review');
	});

	it('replaces spaces with hyphens', () => {
		expect(normalizeSkillName('My Great Skill')).toBe('my-great-skill');
	});

	it('removes special characters', () => {
		expect(normalizeSkillName('Code Review!')).toBe('code-review');
	});

	it('collapses consecutive hyphens', () => {
		expect(normalizeSkillName('code---review')).toBe('code-review');
	});

	it('strips leading/trailing hyphens', () => {
		expect(normalizeSkillName('-code-review-')).toBe('code-review');
	});

	it('handles single character', () => {
		expect(normalizeSkillName('a')).toBe('a');
	});

	it('max length preserved', () => {
		const longName = 'a'.repeat(100);
		const normalized = normalizeSkillName(longName);
		expect(normalized.length).toBe(100);
	});
});

describe('validateSkillName', () => {
	it('returns null for valid names', () => {
		expect(validateSkillName('code-review')).toBeNull();
		expect(validateSkillName('a')).toBeNull();
		expect(validateSkillName('skill123')).toBeNull();
	});

	it('rejects empty names', () => {
		expect(validateSkillName('')).toBe('Name is required');
		expect(validateSkillName('   ')).toBe('Name is required');
	});

	it('rejects names too long', () => {
		expect(validateSkillName('a'.repeat(65))).toBe('Name must be ≤ 64 characters');
	});

	it('rejects uppercase names', () => {
		expect(validateSkillName('Code-Review')).not.toBeNull();
	});

	it('rejects names with leading hyphens', () => {
		expect(validateSkillName('-code')).not.toBeNull();
	});

	it('rejects names with trailing hyphens', () => {
		expect(validateSkillName('code-')).not.toBeNull();
	});

	it('rejects names with consecutive hyphens', () => {
		expect(validateSkillName('code--review')).not.toBeNull();
	});

	it('rejects names with special characters', () => {
		expect(validateSkillName('code@review')).not.toBeNull();
	});

	it('accepts alphanumeric with hyphens', () => {
		expect(validateSkillName('my-skill-123')).toBeNull();
	});
});

describe('validateSkillDescription', () => {
	it('returns null for valid descriptions', () => {
		expect(validateSkillDescription('A good description')).toBeNull();
	});

	it('rejects empty descriptions', () => {
		expect(validateSkillDescription('')).toBe('Description is required');
		expect(validateSkillDescription('   ')).toBe('Description is required');
	});

	it('rejects descriptions too long', () => {
		expect(validateSkillDescription('a'.repeat(1025))).toBe(
			'Description must be ≤ 1024 characters'
		);
	});

	it('accepts descriptions at max length', () => {
		expect(validateSkillDescription('a'.repeat(1024))).toBeNull();
	});
});

describe('skillToMarkdown - Agent Skills compatibility', () => {
	it('does not include private fields like origin in output', () => {
		const skill = {
			name: 'test-skill',
			description: 'Test skill',
			content: 'Some content',
			origin: 'project' as const,
			path: '/etc/skill/test-skill/SKILL.md'
		};
		const md = skillToMarkdown(skill);
		expect(md).not.toContain('origin:');
		expect(md).not.toContain('path:');
	});

	it('normalizes name in output', () => {
		const skill = {
			name: 'Code Review Assistant',
			description: 'Reviews code',
			content: 'Review this'
		};
		const md = skillToMarkdown(skill);
		expect(md).toContain('name: code-review-assistant');
	});

	it('includes optional fields when provided', () => {
		const skill = {
			name: 'test',
			description: 'Test',
			content: 'Content',
			license: 'MIT',
			compatibility: 'Node.js >= 18',
			disableModelInvocation: true
		};
		const md = skillToMarkdown(skill);
		expect(md).toContain('license: MIT');
		expect(md).toContain('compatibility: Node.js >= 18');
		expect(md).toContain('disable-model-invocation: true');
	});

	it('round-trips with Agent Skills spec fields', () => {
		const original = {
			name: 'code-review',
			description: 'Reviews code for quality',
			content: 'Review the code',
			license: 'MIT',
			compatibility: 'Node.js >= 18',
			disableModelInvocation: false
		};
		const md = skillToMarkdown(original);
		const parsed = markdownToSkill(md);
		expect(parsed.name).toBe('code-review');
		expect(parsed.description).toBe('Reviews code for quality');
		expect(parsed.content).toBe('Review the code');
	});
});

describe('composeSystemPromptWithAlwaysOnSkills', () => {
	it('returns base prompt unchanged when no always-on skills', () => {
		const out = composeSystemPromptWithAlwaysOnSkills('base prompt', []);
		expect(out.text).toBe('base prompt');
		expect(out.skills).toEqual([]);
	});

	it('emits only the skill content (no headers, no XML wrapping)', () => {
		const out = composeSystemPromptWithAlwaysOnSkills('', [
			{ id: 'a', name: 'a-name', content: 'be funny' },
			{ id: 'b', name: 'b-name', content: 'talk like a pirate' }
		]);
		// No "### Skill: <name>" header — the system message body must be
		// a byte-for-byte match of the source skill contents so the
		// "Sync from library" flow can compare against the row.
		expect(out.text).toBe('be funny\n\ntalk like a pirate');
		expect(out.text).not.toContain('### Skill:');
		expect(out.text).not.toContain('a-name');
		expect(out.text).not.toContain('b-name');
		expect(out.skills.length).toBe(2);
	});

	it('appends after the base prompt when one is supplied', () => {
		const out = composeSystemPromptWithAlwaysOnSkills('You are helpful.', [
			{ id: 'a', name: 'clown', content: 'be funny' }
		]);
		expect(out.text).toBe('You are helpful.\n\nbe funny');
	});

	it('trims trailing whitespace from each skill body', () => {
		const out = composeSystemPromptWithAlwaysOnSkills('', [
			{ id: 'a', name: 'clown', content: '  be funny  \n\n' }
		]);
		expect(out.text).toBe('be funny');
	});

	it('skips skills with empty content', () => {
		const out = composeSystemPromptWithAlwaysOnSkills('', [
			{ id: 'a', name: 'empty', content: '' },
			{ id: 'b', name: 'real', content: 'be useful' }
		]);
		expect(out.text).toBe('be useful');
		expect(out.skills.length).toBe(1);
		expect(out.skills[0].name).toBe('real');
	});

	it('skips skills with disableModelInvocation=true', () => {
		const out = composeSystemPromptWithAlwaysOnSkills('', [
			{ id: 'a', name: 'silent', content: 'hidden', disableModelInvocation: true },
			{ id: 'b', name: 'active', content: 'visible' }
		]);
		expect(out.text).toBe('visible');
		expect(out.skills.length).toBe(1);
	});

	it('contains no XML wrapping, no llama-ui markers, no skill name', () => {
		const out = composeSystemPromptWithAlwaysOnSkills('', [
			{ id: 'a', name: 'zzzqxlm', content: 'be funny' }
		]);
		expect(out.text).not.toContain('<skill');
		expect(out.text).not.toContain('<!-- llama-ui');
		expect(out.text).not.toContain('location=');
		expect(out.text).not.toContain('zzzqxlm');
	});

	it('falls back to base prompt when every skill is filtered out', () => {
		const out = composeSystemPromptWithAlwaysOnSkills('base prompt', [
			{ id: 'a', name: 'empty', content: '' },
			{ id: 'b', name: 'silent', content: 'hidden', disableModelInvocation: true }
		]);
		expect(out.text).toBe('base prompt');
		expect(out.skills).toEqual([]);
	});
});
