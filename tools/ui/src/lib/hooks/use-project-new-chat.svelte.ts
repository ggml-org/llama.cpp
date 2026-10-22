import { conversationsStore } from '$lib/stores';
import type { Project } from '$lib/types';

/**
 * Starts the screen from the new-chat state with the project folder picked
 * on the form, so the next chat sent from its composer joins the project.
 *
 * @param getProject - The project, undefined until projects are loaded
 */
export function useProjectNewChat(getProject: () => Project | undefined) {
	let seededProjectId: string | null = null;

	$effect(() => {
		const project = getProject();

		if (!project || seededProjectId === project.id) return;

		seededProjectId = project.id;

		conversationsStore.clearActiveConversation();
		void conversationsStore.preferences.setCwd(project.cwd);
		// the composer's project picker may still point the chat somewhere else
		conversationsStore.preferences.pendingProjectId = project.id;
	});
}
