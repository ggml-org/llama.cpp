/**
 *
 * PROJECTS
 *
 * A project groups chats under shared instructions, context and a working
 * directory. It is configured in one dialog and shown as a sidebar section.
 *
 */

/**
 * **DialogProject** - Create or edit a project
 *
 * Name, goal, working directory, instructions and context in one form. When
 * editing, the footer also deletes the project. Opened through
 * `uiStore.openProjectDialog`.
 */
export { default as DialogProject } from './DialogProject.svelte';

/**
 * **DialogProjectPicker** - Pick a project for a chat on a phone
 *
 * The phone's stand-in for `ProjectSubmenu`: a searchable list that can also
 * create a project by name. Opened through `uiStore.openProjectPicker`.
 */
export { default as DialogProjectPicker } from './DialogProjectPicker.svelte';

/**
 * **ProjectCard** - One project on the Projects page
 *
 * Opens the project dialog on click; its right-click menu and "more actions"
 * dropdown edit, pin or delete the project.
 */
export { default as ProjectCard } from './ProjectCard.svelte';

/**
 * **ProjectContextList** - Context items of a project draft
 *
 * Uploads files, references server files or MCP resources, and removes items.
 */
export { default as ProjectContextList } from './ProjectContextList.svelte';

/**
 * **ProjectFolderPicker** - Working directory row of a project draft
 */
export { default as ProjectFolderPicker } from './ProjectFolderPicker.svelte';

/**
 * **ProjectPrefix** - Project icon, name and slash before a chat title
 */
export { default as ProjectPrefix } from './ProjectPrefix.svelte';

/**
 * **ProjectSubmenu** - "Add to project" flyout of a chat menu
 *
 * Searchable project list inside a dropdown or context menu, with a New
 * project entry that opens the project dialog and joins the project it creates.
 */
export { default as ProjectSubmenu } from './ProjectSubmenu.svelte';

/** **ProjectActions** - Pin and menu of the project page's title */
export { default as ProjectActions } from './ProjectActions.svelte';

/** **ProjectBreadcrumb** - Part of the project page */
export { default as ProjectBreadcrumb } from './ProjectBreadcrumb.svelte';

/** **ProjectContext** - Part of the project page */
export { default as ProjectContext } from './ProjectContext.svelte';

/** **ProjectConversations** - Part of the project page */
export { default as ProjectConversations } from './ProjectConversations.svelte';

/** **ProjectHeader** - Part of the project page */
export { default as ProjectHeader } from './ProjectHeader.svelte';

/** **ProjectInstructions** - Part of the project page */
export { default as ProjectInstructions } from './ProjectInstructions.svelte';

/** **ProjectIntro** - Part of the project page */
export { default as ProjectIntro } from './ProjectIntro.svelte';

/** **ProjectOverview** - Part of the project page */
export { default as ProjectOverview } from './ProjectOverview.svelte';

/** **ProjectPanels** - Part of the project page */
export { default as ProjectPanels } from './ProjectPanels.svelte';

/** **ProjectSection** - Part of the project page */
export { default as ProjectSection } from './ProjectSection.svelte';
