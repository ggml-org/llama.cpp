/**
 *
 * Manage
 *
 * Chrome wrapper for "manage X" style pages (MCP servers, prompts, etc.).
 * Renders a sticky page header with title and optional icon, places the
 * primary action on the right on desktop and as a fixed bottom-right
 * button on mobile.
 *
 * @example
 * ```svelte
 * <ManageLayout title="MCP Servers">
 *   {#snippet icon()}
 *     <McpLogo class="h-5 w-5 md:h-6 md:w-6" />
 *   {/snippet}
 *
 *   {#snippet actions()}
 *     <Button onclick={openAddServer}>Add New Server</Button>
 *   {/snippet}
 *
 *   ...page content...
 * </ManageLayout>
 * ```
 */
export { default as ManageLayout } from './ManageLayout.svelte';
