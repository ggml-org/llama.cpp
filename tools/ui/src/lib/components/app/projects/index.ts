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
 * editing, the footer also hides or deletes the project. Opened through
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
