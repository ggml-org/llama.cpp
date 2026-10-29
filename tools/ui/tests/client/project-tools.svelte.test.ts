import { AttachmentType, BuiltInTool, ProjectContextKind } from '$lib/enums';
import { DatabaseService } from '$lib/services/database.service';
import { ProjectToolsService } from '$lib/services/project-tools.service';
import { projectsStore } from '$lib/stores/projects.svelte';
import { afterEach, describe, expect, it } from 'vitest';

afterEach(async () => {
	for (const project of [...projectsStore.projects]) await projectsStore.remove(project.id);

	const rows = await DatabaseService.getAllProjects();

	await Promise.all(rows.map((project) => DatabaseService.deleteProject(project.id)));
});

async function newProject(name = 'llama-ui') {
	const project = await projectsStore.create({
		contextItems: [],
		cwd: null,
		description: '',
		instructions: '',
		name
	});

	if (!project) throw new Error('project not created');

	return project;
}

/**
 * The agent's project tools act on the chat's own project by default and
 * only change the fields a call names.
 */
describe('ProjectToolsService', () => {
	it('lists projects and reads one by id', async () => {
		const project = await newProject();
		const list = await ProjectToolsService.executeTool(BuiltInTool.BROWSER_LIST_PROJECTS, {});

		expect(JSON.parse(list.content)).toEqual([
			{ chats: 0, goal: '', id: project.id, name: 'llama-ui' }
		]);

		const read = await ProjectToolsService.executeTool(BuiltInTool.BROWSER_GET_PROJECT, {
			project_id: project.id
		});

		expect(JSON.parse(read.content)).toMatchObject({ instructions: '', name: 'llama-ui' });
	});

	it("updates the chat's project, appending to its instructions", async () => {
		const project = await newProject();

		await projectsStore.update(project.id, { instructions: 'Answer briefly.' });

		const result = await ProjectToolsService.executeTool(
			BuiltInTool.BROWSER_UPDATE_PROJECT,
			{ append_instructions: 'Prefer Svelte 5 runes.', goal: 'Polish the UI', name: 'llama web' },
			project.id
		);

		expect(result.isError).toBe(false);
		expect(projectsStore.get(project.id)).toMatchObject({
			description: 'Polish the UI',
			instructions: 'Answer briefly.\nPrefer Svelte 5 runes.',
			name: 'llama web'
		});
	});

	it('adds a server file reference and a text note to the context', async () => {
		const project = await newProject();

		await ProjectToolsService.executeTool(
			BuiltInTool.BROWSER_ADD_PROJECT_CONTEXT,
			{ path: '/srv/repo/README.md' },
			project.id
		);
		await ProjectToolsService.executeTool(
			BuiltInTool.BROWSER_ADD_PROJECT_CONTEXT,
			{ note_content: 'Ship on Friday.', note_name: 'decisions.md' },
			project.id
		);

		expect(projectsStore.get(project.id)?.contextItems).toMatchObject([
			{ kind: ProjectContextKind.FILE, name: 'README.md', path: '/srv/repo/README.md' },
			{
				extra: { content: 'Ship on Friday.', name: 'decisions.md', type: AttachmentType.TEXT },
				kind: ProjectContextKind.ATTACHMENT,
				name: 'decisions.md'
			}
		]);
	});

	it('asks for a project outside of a project chat', async () => {
		await newProject();

		const result = await ProjectToolsService.executeTool(BuiltInTool.BROWSER_GET_PROJECT, {});

		expect(result.isError).toBe(true);
		expect(result.content).toContain('list_projects');
	});
});
