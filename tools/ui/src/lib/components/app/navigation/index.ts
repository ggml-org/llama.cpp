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
 * Single conversation item in sidebar. Displays conversation title (truncated),
 * last message preview, and timestamp. Shows context menu on right-click with
 * rename and delete options. Highlights when active (matches current route).
 * Handles click to navigate and keyboard accessibility.
 */
export { default as SidebarNavigationConversationItem } from './SidebarNavigation/SidebarNavigationConversationItem.svelte';

/**
 * **SidebarNavigationSelectionBar** - Bulk action toolbar for selection mode
 *
 * Rendered above the conversation list when the sidebar enters selection mode.
 * Hosts a master checkbox (with select-all / clear-all semantics over the
 * currently-visible items), a selected-count caption, and bulk actions for
 * pin/unpin, export, and delete. Delete uses
 * {@link DialogConfirmation} before invoking the bulk store method.
 *
 * Pure-presentational; all operations are delegated via callbacks so the
 * sidebar owns selection state and persistence.
 *
 * @example
 * ```svelte
 * <SidebarNavigationSelectionBar
 *   selectedCount={selectedIds.size}
 *   visibleCount={visibleConversations.length}
 *   allVisibleSelected={...}
 *   someVisibleSelected={...}
 *   someSelectedPinned={...}
 *   onSelectAllToggle={toggleSelectAll}
 *   onBulkPinToggle={handleBulkPin}
 *   onBulkExport={handleBulkExport}
 *   onBulkDelete={handleBulkDelete}
 *   onClose={exitSelectionMode}
 * />
 * ```
 */
export { default as SidebarNavigationSelectionBar } from './SidebarNavigation/SidebarNavigationSelectionBar.svelte';

/**
 * **SidebarNavigationConversationList** - Grouped conversation list
 *
 * Pure-presentational list of conversations. Splits items into a Pinned
 * section (when not in search mode) and a Recent Conversations / Search
 * Results section with the unpinned items. Item selection, edit, delete,
 * and stop-generation are delegated to the caller via callbacks.
 *
 * @example
 * ```svelte
 * <SidebarNavigationConversationList
 *   {filteredConversations}
 *   {currentChatId}
 *   {isSearchModeActive}
 *   {searchQuery}
 *   onSelect={...}
 *   onEdit={...}
 *   onDelete={...}
 *   onStop={...}
 * />
 * ```
 */
export { default as SidebarNavigationConversationList } from './SidebarNavigation/SidebarNavigationConversationList.svelte';
export { default as SidebarNavigationActions } from './SidebarNavigation/SidebarNavigationActions.svelte';

/**
 * **SidebarNavigationSearchResults** - Filtered conversation list for search.
 *
 * Pure-presentational rendering of the search-mode subtree: "Search results"
 * header, the matching items rendered through {@link SidebarNavigationConversationItem},
 * and contextual empty-state messages. Used both inline inside
 * {@link SidebarNavigationConversationList} (when search mode is active in the
 * sidebar) and as the body of the mobile `/search` route.
 *
 * The caller is expected to provide an already-filtered list via
 * `filteredConversations` and a `searchQuery` for the empty-state messages.
 *
 * @example
 * ```svelte
 * <SidebarNavigationSearchResults
 *   {searchQuery}
 *   {filteredConversations}
 *   {currentChatId}
 *   onSelect={...}
 *   onEdit={...}
 *   onDelete={...}
 *   onStop={...}
 * />
 * ```
 */
export { default as SidebarNavigationSearchResults } from './SidebarNavigation/SidebarNavigationSearchResults.svelte';

/**
 * Search input for filtering conversations in sidebar. Filters conversation
 * list by title as user types. Shows clear button when query is not empty.
 * Integrated into sidebar header with proper styling.
 */
export { default as SidebarNavigationSearch } from './SidebarNavigation/SidebarNavigationSearch.svelte';
