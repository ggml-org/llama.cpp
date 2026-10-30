import { ProjectContextKind } from '$lib/enums';
import { buildProjectSystemPrompt } from '$lib/utils/project-prompt';
import { describe, expect, it } from 'vitest';

describe('buildProjectSystemPrompt', () => {
	it('passes the instructions through as written', () => {
		const instructions = 'You are a terse reviewer.\n\n- Answer in English\n- Cite files';

		expect(buildProjectSystemPrompt({ contextItems: [], instructions, name: 'Review' })).toBe(
			instructions
		);
	});

	it('adds the context list after the instructions', () => {
		expect(
			buildProjectSystemPrompt({
				contextItems: [
					{ id: '1', kind: ProjectContextKind.FILE, name: 'a.md', path: '/repo/a.md' }
				],
				instructions: 'Be brief.',
				name: 'Docs'
			})
		).toBe(
			'Be brief.\n\nContext the user added to the project "Docs":\n- /repo/a.md (server file)'
		);
	});

	it('is empty for a project with neither', () => {
		expect(buildProjectSystemPrompt({ contextItems: [], instructions: '  ', name: 'X' })).toBe('');
		expect(buildProjectSystemPrompt(undefined)).toBe('');
	});
});
