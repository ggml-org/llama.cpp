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
