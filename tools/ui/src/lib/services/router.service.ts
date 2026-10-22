/**
 * RouterService - Builds app route paths
 *
 * Returns chat route strings from a single source of truth (ROUTES). No state.
 */

import { ROUTES } from '$lib/constants';

export class RouterService {
	static chat(id: string): string {
		return `${ROUTES.CHAT}/${id}`;
	}

	static project(id: string): string {
		return `${ROUTES.PROJECTS}/${id}`;
	}

	/** A chat opened inside its project, with the project's tabs. */
	static projectChat(projectId: string, id: string): string {
		return `${RouterService.project(projectId)}/chat/${id}`;
	}

	/** The new-chat screen of a project; the chat it starts joins the project. */
	static projectNewChat(projectId: string): string {
		return `${RouterService.project(projectId)}/chat/new`;
	}
}
