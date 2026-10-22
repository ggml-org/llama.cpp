/** Sentinel tab id for the bare `#/` new-chat screen */
export const NEW_CHAT_TAB_ID = 'new-chat';

/** Route target of a project's home page, which never gets a tab */
export const PROJECT_HOME_TAB_ID = 'project-home';

/** Label shown for the new-chat sentinel tab. */
export const NEW_CHAT_LABEL = 'New chat';

/** Fallback label for conversations without an auto-generated title. */
export const UNNAMED_CHAT_LABEL = 'Chat';

/**
 * Tab bar max width so it stays clear of the sidebar strip. The expanded strip
 * is `--sidebar-width` wide (user resizable) and the collapsed one `md:w-12`;
 * these hold the fully tuned `max-w-[calc(100vw-?)]` classes so the offset has
 * a single source.
 */
export const CHAT_TABS_MAX_WIDTH = {
	COLLAPSED_SIDEBAR: 'max-w-[calc(100vw-5rem)]',
	EXPANDED_SIDEBAR: 'max-w-[calc(100vw-var(--sidebar-width)-2rem)]'
} as const;
