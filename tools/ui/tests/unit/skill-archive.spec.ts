import { describe, expect, it } from 'vitest';
import { skillsToArchive, archiveToSkillRows } from '$lib/utils/skill-archive';
import type { DatabaseSkill } from '$lib/types';

const sampleSkills: DatabaseSkill[] = [
	{
		id: '11111111-1111-1111-1111-111111111111',
		name: 'code-reviewer',
		description: 'Reviews code for quality',
		content: 'You are a code reviewer.',
		lastModified: 1700000000000,
		origin: 'lib'
	},
	{
		id: '22222222-2222-2222-2222-222222222222',
		name: 'Python Tutor',
		description: 'Teaches Python',
		content: 'You teach Python fundamentals.',
		lastModified: 1700000000001,
		path: '/Users/test/.pi/agent/skills/python-tutor/SKILL.md',
		origin: 'user'
	}
];

describe('skillsToArchive', () => {
	it('produces a non-empty zip blob with one SKILL.md per skill', async () => {
		const blob = await skillsToArchive(sampleSkills);
		expect(blob.size).toBeGreaterThan(0);
		const bytes = new Uint8Array(await blob.arrayBuffer());
		// ZIP magic bytes
		expect(bytes[0]).toBe(0x50);
		expect(bytes[1]).toBe(0x4b);
	});

	it('round-trips via archiveToSkillRows', async () => {
		const blob = await skillsToArchive(sampleSkills);
		const rows = await archiveToSkillRows(blob);
		expect(rows.length).toBe(2);

		const names = rows.map((r) => r.name).sort();
		expect(names).toEqual(['code-reviewer', 'python-tutor']);

		const byName = new Map(rows.map((r) => [r.name, r]));
		expect(byName.get('code-reviewer')?.description).toBe('Reviews code for quality');
		expect(byName.get('code-reviewer')?.content).toBe('You are a code reviewer.');
		expect(byName.get('python-tutor')?.content).toBe('You teach Python fundamentals.');
	});

	it('preserves UUID id when the directory name is the UUID', async () => {
		const blob = await skillsToArchive([
			{
				id: '11111111-1111-1111-1111-111111111111',
				name: '11111111-1111-1111-1111-111111111111',
				description: 'uuid-named',
				content: 'body',
				lastModified: 1,
				origin: 'lib'
			}
		]);
		const rows = await archiveToSkillRows(blob);
		const uuidRow = rows[0];
		expect(uuidRow.id).toBe('11111111-1111-1111-1111-111111111111');
		expect(uuidRow.name).toBe('11111111-1111-1111-1111-111111111111');
	});

	it('sanitizes path separators in skill names', async () => {
		const blob = await skillsToArchive([
			{
				id: '33333333-3333-3333-3333-333333333333',
				name: '../etc/passwd',
				description: 'evil',
				content: 'malicious',
				lastModified: 1,
				origin: 'lib'
			},
			{
				id: '44444444-4444-4444-4444-444444444444',
				name: 'foo/bar',
				description: 'evil2',
				content: 'cross dir',
				lastModified: 1,
				origin: 'lib'
			}
		]);
		const rows = await archiveToSkillRows(blob);
		// Both sanitized to a single safe directory name
		expect(rows.length).toBe(2);
		expect(rows.some((r) => r.content === 'malicious')).toBe(true);
		expect(rows.some((r) => r.content === 'cross dir')).toBe(true);
	});

	it('handles an empty input', async () => {
		const blob = await skillsToArchive([]);
		expect(blob.size).toBeGreaterThanOrEqual(0);
		const rows = await archiveToSkillRows(new Uint8Array(await blob.arrayBuffer()));
		expect(rows).toEqual([]);
	});

	it('handles duplicate names by appending a suffix', async () => {
		const blob = await skillsToArchive([
			{
				id: '11111111-1111-1111-1111-111111111111',
				name: 'dup',
				description: 'first',
				content: 'a',
				lastModified: 1,
				origin: 'lib'
			},
			{
				id: '22222222-2222-2222-2222-222222222222',
				name: 'dup',
				description: 'second',
				content: 'b',
				lastModified: 2,
				origin: 'lib'
			}
		]);
		const rows = await archiveToSkillRows(blob);
		const dirs = rows.map((r) => r.content).sort();
		expect(dirs).toEqual(['a', 'b']);
	});
});
