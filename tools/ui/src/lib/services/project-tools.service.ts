/**
 * ProjectToolsService - the browser tools that read and edit projects
 *
 * Lets the agent list projects, read one, rename it, set its goal, edit its
 * instructions and add context to it. Every call still passes the tool
 * permission prompt first; the agentic store dispatches here.
 */

import { AttachmentType, BuiltInTool, ProjectContextKind } from '$lib/enums';
// direct imports between stores, not via the barrel, to avoid circular deps
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import { projectsStore } from '$lib/stores/projects.svelte';
import type { Project, ProjectContextItem, ProjectDraft, ToolExecutionResult } from '$lib/types';
import { lastPathSegment } from '$lib/utils/path-display';

function ok(value: unknown): ToolExecutionResult {
	return { content: JSON.stringify(value), isError: false };
}

function error(message: string): ToolExecutionResult {
	return { content: `Error: ${message}`, isError: true };
}

function text(value: unknown): string | undefined {
	return typeof value === 'string' ? value : undefined;
}

function summary(project: Project) {
	return {
		chats: conversationsStore.conversations.filter((c) => c.projectId === project.id).length,
		goal: project.description,
		id: project.id,
		name: project.name
	};
}

function details(project: Project) {
	return {
		...summary(project),
		context: project.contextItems.map((item) => ({
			kind: item.kind,
			name: item.name,
			path: item.path
		})),
		instructions: project.instructions,
		working_directory: project.cwd
	};
}

export class ProjectToolsService {
	/**
	 * @param name - Tool name, one of the project tools
	 * @param args - Parsed tool call arguments
	 * @param currentProjectId - Project of the chat the call comes from, if any
	 */
	static async executeTool(
		name: string,
		args: Record<string, unknown>,
		currentProjectId?: string | null
	): Promise<ToolExecutionResult> {
		if (name === BuiltInTool.BROWSER_LIST_PROJECTS) {
			return ok(projectsStore.projects.map(summary));
		}

		const projectId = text(args.project_id) ?? currentProjectId ?? undefined;

		if (!projectId) {
			return error('This chat is not in a project: pass project_id, from list_projects.');
		}

		if (!projectsStore.get(projectId)) return error(`No project with id "${projectId}".`);

		switch (name) {
			case BuiltInTool.BROWSER_GET_PROJECT:
				return ok(details(projectsStore.get(projectId)!));

			case BuiltInTool.BROWSER_UPDATE_PROJECT:
				return this.update(projectId, args);

			case BuiltInTool.BROWSER_ADD_PROJECT_CONTEXT:
				return this.addContext(projectId, args);

			default:
				return error(`Unknown project tool "${name}".`);
		}
	}

	private static async addContext(
		projectId: string,
		args: Record<string, unknown>
	): Promise<ToolExecutionResult> {
		const path = text(args.path)?.trim();
		const noteName = text(args.note_name)?.trim();
		const noteContent = text(args.note_content);

		let item: Omit<ProjectContextItem, 'id'>;

		if (path) {
			item = { kind: ProjectContextKind.FILE, name: lastPathSegment(path), path };
		} else if (noteName && noteContent) {
			item = {
				extra: { content: noteContent, name: noteName, type: AttachmentType.TEXT },
				kind: ProjectContextKind.ATTACHMENT,
				name: noteName,
				size: new Blob([noteContent]).size
			};
		} else {
			return error('Pass either path, or note_name with note_content.');
		}

		const { contextItems } = projectsStore.get(projectId)!;

		// the store gives the new item its id
		await projectsStore.update(projectId, {
			contextItems: [...contextItems, { ...item, id: '' }]
		});

		return ok(details(projectsStore.get(projectId)!));
	}

	private static async update(
		projectId: string,
		args: Record<string, unknown>
	): Promise<ToolExecutionResult> {
		const name = text(args.name)?.trim();
		const goal = text(args.goal);
		const instructions = text(args.instructions);
		const append = text(args.append_instructions);

		if (name === undefined && goal === undefined && instructions === undefined && !append) {
			return error('Nothing to update: pass name, goal, instructions or append_instructions.');
		}

		const changes: Partial<ProjectDraft> = {};

		if (name) changes.name = name;

		if (goal !== undefined) changes.description = goal;

		if (instructions !== undefined || append) {
			const current = instructions ?? projectsStore.get(projectId)!.instructions;

			changes.instructions = append
				? [current.trim(), append.trim()].filter(Boolean).join('\n')
				: current;
		}

		await projectsStore.update(projectId, changes);

		return ok(details(projectsStore.get(projectId)!));
	}
}
