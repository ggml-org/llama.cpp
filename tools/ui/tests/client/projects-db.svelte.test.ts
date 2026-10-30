import { ProjectContextKind } from '$lib/enums';
import { DatabaseService } from '$lib/services/database.service';
import type { Project } from '$lib/types';
import { afterEach, describe, expect, it } from 'vitest';

function makeProject(id: string): Project {
	return {
		contextItems: [],
		createdAt: 0,
		cwd: null,
		description: '',
		id,
		instructions: '',
		name: `Project ${id}`,
		updatedAt: 0
	};
}

afterEach(async () => {
	const projects = await DatabaseService.getAllProjects();

	await Promise.all(projects.map((project) => DatabaseService.deleteProject(project.id)));
});

/**
 * Projects live in their own table, so their rows must survive a write/read
 * round trip with the nested context files and the folder they carry.
 */
describe('DatabaseService projects', () => {
	it('round-trips a project with its context items', async () => {
		const project = {
			...makeProject('a'),
			contextItems: [
				{ id: 'file-1', kind: ProjectContextKind.ATTACHMENT, name: 'spec.md', size: 10 }
			],
			cwd: '/home/user/work',
			description: 'Ship the projects UI'
		};

		await DatabaseService.createProject(project);

		expect(await DatabaseService.getAllProjects()).toEqual([project]);
	});

	it('applies partial updates without dropping other fields', async () => {
		await DatabaseService.createProject(makeProject('a'));

		await DatabaseService.updateProject('a', { name: 'Renamed', updatedAt: 5 });

		const [project] = await DatabaseService.getAllProjects();

		expect(project.name).toBe('Renamed');
		expect(project.updatedAt).toBe(5);
	});

	it('returns projects newest first', async () => {
		await DatabaseService.createProject({ ...makeProject('a'), updatedAt: 1 });
		await DatabaseService.createProject({ ...makeProject('b'), updatedAt: 2 });

		expect((await DatabaseService.getAllProjects()).map((project) => project.id)).toEqual([
			'b',
			'a'
		]);
	});

	it('deletes a project', async () => {
		await DatabaseService.createProject(makeProject('a'));

		await DatabaseService.deleteProject('a');

		expect(await DatabaseService.getAllProjects()).toEqual([]);
	});
});
