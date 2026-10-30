/**
 * projectsStore - Projects, their instructions and context files
 *
 * Mirrors the row-per-project table through DatabaseService, like
 * conversationsStore: every mutation awaits the row write, then applies the
 * change to the loaded list.
 */

import { browser } from '$app/environment';
import { PROJECT_NAME_MAX_LENGTH } from '$lib/constants';
import { ConversationTransferService } from '$lib/services/conversation-transfer.service';
import { DatabaseService } from '$lib/services/database.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import type { Project, ProjectDraft } from '$lib/types';
import { uuid } from '$lib/utils';
import { toast } from 'svelte-sonner';

/** Trims the text fields of a draft and gives new context items their ids. */
function normalize(draft: ProjectDraft): ProjectDraft {
	return {
		contextItems: draft.contextItems.map((item) => (item.id ? item : { ...item, id: uuid() })),
		cwd: draft.cwd,
		description: draft.description.trim(),
		instructions: draft.instructions.trim(),
		name: draft.name.trim().slice(0, PROJECT_NAME_MAX_LENGTH)
	};
}

class ProjectsStore {
	/** Whether the list was read from the database at least once. */
	isInitialized = $state(false);

	/** All projects, newest first. */
	projects = $state<Project[]>([]);

	private initPromise: Promise<void> | null = null;

	/**
	 * Creates a project and puts it at the top of the list.
	 *
	 * @param draft - What the project dialog collected
	 * @returns The created project, or undefined when the name is empty
	 */
	async create(draft: ProjectDraft): Promise<Project | undefined> {
		const name = draft.name.trim().slice(0, PROJECT_NAME_MAX_LENGTH);

		if (!name) return undefined;

		const now = Date.now();
		const project: Project = {
			...normalize(draft),
			createdAt: now,
			id: uuid(),
			name,
			updatedAt: now
		};

		try {
			await DatabaseService.createProject(project);
		} catch (error) {
			console.error('Failed to create project:', error);

			return undefined;
		}

		this.projects = [project, ...this.projects];

		return project;
	}

	/**
	 * Downloads the project and the conversations it holds as a `.zip`.
	 *
	 * @param id - Project ID
	 */
	async exportArchive(id: string): Promise<void> {
		const project = this.get(id);

		if (!project) return;

		try {
			const convIds = conversationsStore.conversations
				.filter((conversation) => conversation.projectId === project.id)
				.map((conversation) => conversation.id);
			const exported = await conversationsStore.getConversationsForExport(convIds);

			ConversationTransferService.downloadProjectArchive(project, exported);

			toast.success(
				exported.length === 1
					? 'Project and 1 conversation exported'
					: `Project and ${exported.length} conversations exported`
			);
		} catch (error) {
			console.error('Failed to export project:', error);
			toast.error('Failed to export project');
		}
	}

	/** Takes the projects of a deleted group out of it; the database did so already. */
	forgetGroup(groupId: string): void {
		this.projects = this.projects.map((project) =>
			project.groupId === groupId ? { ...project, groupId: undefined } : project
		);
	}

	get(id: string): Project | undefined {
		return this.projects.find((project) => project.id === id);
	}

	/**
	 * Reads every project from the database.
	 * Safe to call multiple times: concurrent callers share a single run,
	 * and a failed run can be retried by calling again.
	 */
	initialize(): Promise<void> {
		if (!browser) return Promise.resolve();

		if (this.initPromise) return this.initPromise;

		this.initPromise = (async () => {
			try {
				this.projects = await DatabaseService.getAllProjects();
				this.isInitialized = true;
			} catch (error) {
				console.error('Failed to load projects:', error);
				this.initPromise = null;
			}
		})();

		return this.initPromise;
	}

	/**
	 * Deletes a project. Its chats stay, outside any project.
	 *
	 * @param id - Project ID
	 */
	async remove(id: string): Promise<void> {
		try {
			await DatabaseService.deleteProject(id);
		} catch (error) {
			console.error('Failed to delete project:', error);

			return;
		}

		this.projects = this.projects.filter((project) => project.id !== id);

		const chats = conversationsStore.conversations.filter((conv) => conv.projectId === id);

		await Promise.all(
			chats.map((conv) => conversationsStore.setConversationProject(conv.id, null))
		);
	}

	/**
	 * Moves the project's sidebar row into a group, or out of one with null.
	 *
	 * @param id - Project ID
	 * @param groupId - Target group, null for none
	 */
	async setGroup(id: string, groupId: string | null): Promise<void> {
		await this.apply(id, { groupId: groupId ?? undefined });
	}

	/** Pins the project's sidebar row, or unpins it. */
	async setPinned(id: string, pinned: boolean): Promise<void> {
		await this.apply(id, { pinned: pinned || undefined });
	}

	/**
	 * Edits a project: any of its name, goal, folder, instructions and context.
	 * An empty name keeps the current one.
	 *
	 * @param id - Project ID
	 * @param changes - The fields to change
	 */
	async update(id: string, changes: Partial<ProjectDraft>): Promise<void> {
		const project = this.get(id);

		if (!project) return;

		const next = normalize({ ...project, ...changes });

		await this.patch(id, { ...next, name: next.name || project.name });
	}

	/**
	 * Writes changes without touching `updatedAt`, for flags that do not edit content.
	 */
	private async apply(id: string, changes: Partial<Omit<Project, 'id'>>): Promise<void> {
		// state proxies cannot be structured-cloned into IndexedDB
		const plain = $state.snapshot(changes);

		try {
			await DatabaseService.updateProject(id, plain);
		} catch (error) {
			console.error('Failed to update project:', error);

			return;
		}

		this.applyToState(id, plain);
	}

	private applyToState(id: string, changes: Partial<Omit<Project, 'id'>>): void {
		this.projects = this.projects.map((project) =>
			project.id === id ? { ...project, ...changes } : project
		);
	}

	/**
	 * Writes content changes and marks the project as just edited.
	 */
	private async patch(id: string, changes: Partial<Omit<Project, 'id'>>): Promise<void> {
		await this.apply(id, { ...changes, updatedAt: Date.now() });
	}
}

export const projectsStore = new ProjectsStore();
