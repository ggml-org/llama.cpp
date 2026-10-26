import { NEW_CHAT_TAB_ID, PROJECT_HOME_TAB_ID, ROUTES } from '$lib/constants';
import { RouterService } from '$lib/services/router.service';
import type { DatabaseConversation } from '$lib/types';

interface TabRoute {
	params: Partial<Record<string, string>>;
	route: { id: string | null };
}

/** Which tabs a route shows and where they lead. */
export interface TabView {
	/** Tab of the current route; null on a page without one (e.g. a project home). */
	activeId: string | null;
	/** Where to go when the last visible tab closes. */
	fallbackHref: string;
	hrefFor: (id: string) => string;
	/** Visible tab ids, in strip order. */
	ids: string[];
	/** The project of a project view; undefined in the regular view. */
	projectId?: string;
}

const PROJECT_NEW_CHAT_ROUTE = '/(chat)/projects/[projectId]/chat/new';
const PROJECT_NEW_CHAT_PREFIX = `${NEW_CHAT_TAB_ID}:`;

/** Tab id of the new-chat screen of a project, one per project. */
export function projectNewChatTabId(projectId: string): string {
	return PROJECT_NEW_CHAT_PREFIX + projectId;
}

/** Project of a project new-chat tab id, undefined for any other id. */
export function projectOfNewChatTab(id: string): string | undefined {
	return id.startsWith(PROJECT_NEW_CHAT_PREFIX)
		? id.slice(PROJECT_NEW_CHAT_PREFIX.length)
		: undefined;
}

/**
 * Tab target of the current route: a conversation id, a new-chat screen, or
 * a project home (which never gets a tab). Null on routes without tabs.
 */
export function resolveTabTarget(page: TabRoute): string | null {
	const { id, projectId } = page.params;

	if (id) return id;

	if (projectId) {
		return page.route.id === PROJECT_NEW_CHAT_ROUTE
			? projectNewChatTabId(projectId)
			: PROJECT_HOME_TAB_ID;
	}

	return page.route.id === '/(chat)' ? NEW_CHAT_TAB_ID : null;
}

/**
 * Tabs of the current route. The project view (`#/projects/<id>` and its
 * chats under `/chat/...`) shows the chats of its project and keeps them in
 * it, the project home with none of them active; the regular view shows
 * every chat. Both read the same open tabs.
 */
export function resolveTabView(
	page: TabRoute,
	openTabs: string[],
	conversations: DatabaseConversation[]
): TabView {
	const activeId = resolveTabTarget(page);
	const projectId = page.params.projectId;
	const projectOf = (id: string) =>
		projectOfNewChatTab(id) ?? conversations.find((c) => c.id === id)?.projectId;

	if (projectId) {
		return {
			activeId: activeId === PROJECT_HOME_TAB_ID ? null : activeId,
			fallbackHref: RouterService.project(projectId),
			hrefFor: (id) =>
				projectOfNewChatTab(id)
					? RouterService.projectNewChat(projectId)
					: RouterService.projectChat(projectId, id),
			ids: openTabs.filter((id) => projectOf(id) === projectId),
			projectId
		};
	}

	return {
		activeId,
		fallbackHref: ROUTES.START,
		hrefFor: (id) => (id === NEW_CHAT_TAB_ID ? ROUTES.START : RouterService.chat(id)),
		// a project's new-chat tab lives in its project view only
		ids: openTabs.filter((id) => !projectOfNewChatTab(id))
	};
}
