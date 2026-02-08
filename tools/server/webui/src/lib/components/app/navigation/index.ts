/**
 *
 * NAVIGATION & MENUS
 *
 * Components for dropdown menus and action selection.
 *
 */

/**
 * **DropdownMenuSearchable** - Filterable dropdown menu
 *
 * Generic dropdown with search input for filtering options.
 * Uses Svelte snippets for flexible content rendering.
 *
 * **Features:**
 * - Search/filter input with clear on close
 * - Keyboard navigation support
 * - Custom trigger, content, and footer via snippets
 * - Empty state message
 * - Disabled state
 * - Configurable alignment and width
 *
 * @example
 * ```svelte
 * <DropdownMenuSearchable
 *   bind:open
 *   bind:searchValue
 *   placeholder="Search..."
 *   isEmpty={filteredItems.length === 0}
 * >
 *   {#snippet trigger()}<Button>Select</Button>{/snippet}
 *   {#snippet children()}{#each items as item}<Item {item} />{/each}{/snippet}
 * </DropdownMenuSearchable>
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
