/**
 *
 * NAVIGATION & MENUS
 *
 * Components for dropdown menus and action selection.
 *
 */

/**
 * **DropdownMenuSearchable** - Searchable content for dropdown menus
 *
 * Renders a search input with filtered content area, empty state, and optional footer.
 * Designed to be injected into any dropdown container (DropdownMenu.Content,
 * DropdownMenu.SubContent, etc.) without providing its own Root.
 *
 * **Features:**
 * - Search/filter input
 * - Keyboard navigation support
 * - Custom content and footer via snippets
 * - Empty state message
 *
 * @example
 * ```svelte
 * <DropdownMenu.Root>
 *   <DropdownMenu.Trigger>...</DropdownMenu.Trigger>
 *   <DropdownMenu.Content class="pt-0">
 *     <DropdownMenuSearchable
 *       bind:searchValue
 *       placeholder="Search..."
 *       isEmpty={filteredItems.length === 0}
 *     >
 *       {#each items as item}<Item {item} />{/each}
 *     </DropdownMenuSearchable>
 *   </DropdownMenu.Content>
 * </DropdownMenu.Root>
 * ```
 */
export { default as DropdownMenuSearchable } from './DropdownMenuSearchable.svelte';

/**
 * **DropdownMenuActions** - Multi-action dropdown menu
 *
 * Dropdown menu for multiple action options with icons and shortcuts.
 * Supports destructive variants and keyboard shortcut hints.
 *
 * **Features:**
 * - Configurable trigger icon with tooltip
 * - Action items with icons and labels
 * - Destructive variant styling
 * - Keyboard shortcut display
 * - Separator support between groups
 *
 * @example
 * ```svelte
 * <DropdownMenuActions
 *   triggerIcon={MoreHorizontal}
 *   triggerTooltip="More actions"
 *   actions={[
 *     { icon: Edit, label: 'Edit', onclick: handleEdit },
 *     { icon: Trash, label: 'Delete', onclick: handleDelete, variant: 'destructive' }
 *   ]}
 * />
 * ```
 */
export { default as DropdownMenuActions } from './DropdownMenuActions.svelte';

/**
 * **DesktopIconStrip** - Fixed icon strip for desktop sidebar
 *
 * Vertical icon strip shown on desktop when the sidebar is collapsed.
 * Contains navigation shortcuts for new chat, search, MCP, import/export, and settings.
 */
export { default as DesktopIconStrip } from './DesktopIconStrip.svelte';

/**
 * **SidebarNavigation** - Sidebar with actions menu and conversation list
 *
 * Collapsible sidebar displaying conversation history with search and
 * management actions. Integrates with ShadCN sidebar component for
 * consistent styling and mobile responsiveness.
 *
 * **Architecture:**
 * - Uses ShadCN Sidebar.* components for structure
 * - Fetches conversations from conversationsStore
 * - Manages search state and filtered results locally
 * - Handles conversation CRUD operations via conversationsStore
 *
 * **Navigation:**
 * - Click conversation to navigate to `/chat/[id]`
 * - New chat button navigates to `/` (root)
 * - Active conversation highlighted based on route params
 *
 * **Conversation Management:**
 * - Right-click or menu button for context menu
 * - Rename: Opens inline edit dialog
 * - Delete: Shows confirmation with conversation preview
 * - Delete All: Removes all conversations with confirmation
 *
 * **Features:**
 * - Search/filter conversations by title
 * - Conversation list with message previews (first message truncated)
 * - Active conversation highlighting
 * - Mobile-responsive collapse/expand via ShadCN sidebar
 * - New chat button in header
 * - Settings button opens DialogChatSettings
 *
 * **Exported API:**
 * - `activateSearchMode()` - Focus search input programmatically
 * - `editActiveConversation()` - Open rename dialog for current conversation
 *
 * @example
 * ```svelte
 * <SidebarNavigation bind:this={sidebarRef} />
 * ```
 */
export { default as SidebarNavigation } from './SidebarNavigation/SidebarNavigation.svelte';

/**
 * Action buttons for sidebar header. Contains new chat button, settings button,
 * and delete all conversations button. Manages dialog states for settings and
 * delete confirmation.
 */
export { default as SidebarNavigationActions } from './SidebarNavigation/SidebarNavigationActions.svelte';

/**
 * Single conversation item in sidebar. Displays conversation title (truncated),
 * last message preview, and timestamp. Shows context menu on right-click with
 * rename and delete options. Highlights when active (matches current route).
 * Handles click to navigate and keyboard accessibility.
 */
export { default as SidebarNavigationConversationItem } from './SidebarNavigation/SidebarNavigationConversationItem.svelte';

/**
 * Search input for filtering conversations in sidebar. Filters conversation
 * list by title as user types. Shows clear button when query is not empty.
 * Integrated into sidebar header with proper styling.
 */
export { default as SidebarNavigationSearch } from './SidebarNavigation/SidebarNavigationSearch.svelte';
