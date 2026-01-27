/**
 *
 * MCP (Model Context Protocol)
 *
 * Components for managing MCP server connections and displaying server status.
 * MCP enables agentic workflows by connecting to external tool servers.
 *
 * The MCP system integrates with:
 * - `mcpStore` for server CRUD operations and health checks
 * - `conversationsStore` for per-conversation server enable/disable
 *
 */

/**
 * **McpServersSettings** - MCP servers configuration section
 *
 * Settings section for configuring MCP server connections.
 * Displays server cards with status, tools, and management actions.
 * Used within DialogMcpServersSettings.
 *
 * **Architecture:**
 * - Manages add server form state locally
 * - Delegates server display to McpServerCard components
 * - Integrates with mcpStore for server operations
 * - Shows skeleton loading states during health checks
 *
 * **Features:**
 * - Add new MCP servers by URL with validation
 * - Server cards with connection status indicators
 * - Health check status (connected/disconnected/error)
 * - Tools list per server showing available capabilities
 * - Enable/disable toggle per conversation
 * - Edit/delete server actions
 * - Skeleton loading states during connection
 * - Empty state with helpful message
 *
 * @example
 * ```svelte
 * <McpServersSettings />
 * ```
 */
export { default as McpServersSettings } from './McpServersSettings.svelte';

/**
 * **McpActiveServersAvatars** - Active MCP servers indicator
 *
 * Compact avatar row showing favicons of active MCP servers.
 * Displays up to 3 server icons with "+N" counter for additional servers.
 * Clickable to open MCP settings dialog.
 *
 * **Architecture:**
 * - Filters servers by enabled status and health check
 * - Fetches favicons from server URLs
 * - Integrates with conversationsStore for per-chat server state
 *
 * **Features:**
 * - Overlapping favicon avatars (max 3 visible)
 * - "+N" counter for additional servers
 * - Click handler for settings navigation
 * - Disabled state support
 * - Only shows healthy, enabled servers
 *
 * @example
 * ```svelte
 * <McpActiveServersAvatars
 *   onSettingsClick={() => showMcpSettings = true}
 * />
 * ```
 */
export { default as McpActiveServersAvatars } from './McpActiveServersAvatars.svelte';

/**
 * **McpServerSelector** - Quick MCP server toggle dropdown
 *
 * Compact dropdown for quickly enabling/disabling MCP servers for the current chat.
 * Uses McpActiveServersAvatars as trigger and shows searchable server list with switches.
 *
 * **Architecture:**
 * - Uses DropdownMenuSearchable for searchable dropdown UI
 * - McpActiveServersAvatars as the trigger element
 * - Integrates with conversationsStore for per-chat toggle
 * - Runs health checks on dropdown open
 *
 * **Features:**
 * - Searchable server list by name/URL
 * - Switch toggles matching McpServersSettings behavior
 * - Error state display for unhealthy servers
 * - Footer link to full MCP settings dialog
 *
 * @example
 * ```svelte
 * <McpServerSelector
 *   onSettingsClick={() => showMcpSettings = true}
 * />
 * ```
 */
export { default as McpServerSelector } from './McpServerSelector.svelte';

/**
 * **McpCapabilitiesBadges** - Server capabilities display
 *
 * Displays MCP server capabilities as colored badges.
 * Shows which features the server supports (tools, resources, prompts, etc.).
 *
 * **Features:**
 * - Tools badge (green) - server provides callable tools
 * - Resources badge (blue) - server provides data resources
 * - Prompts badge (purple) - server provides prompt templates
 * - Logging badge (orange) - server supports logging
 * - Completions badge (cyan) - server provides completions
 * - Tasks badge (pink) - server supports task management
 */
export { default as McpCapabilitiesBadges } from './McpCapabilitiesBadges.svelte';

/**
 * **McpConnectionLogs** - Connection log viewer
 *
 * Collapsible panel showing MCP server connection logs.
 * Displays timestamped log entries with level-based styling.
 *
 * **Features:**
 * - Collapsible log list with entry count
 * - Connection time display in milliseconds
 * - Log level icons and color coding
 * - Scrollable log container with max height
 * - Monospace font for log readability
 */
export { default as McpConnectionLogs } from './McpConnectionLogs.svelte';

/**
 * **McpServerForm** - Server URL and headers input form
 *
 * Reusable form for entering MCP server connection details.
 * Used in both add new server and edit server flows.
 *
 * **Features:**
 * - URL input with validation error display
 * - Custom headers key-value pairs editor
 * - Controlled component with change callbacks
 *
 * @example
 * ```svelte
 * <McpServerForm
 *   url={serverUrl}
 *   headers={serverHeaders}
 *   onUrlChange={(v) => serverUrl = v}
 *   onHeadersChange={(v) => serverHeaders = v}
 *   urlError={validationError}
 * />
 * ```
 */
export { default as McpServerForm } from './McpServerForm.svelte';

/**
 * MCP protocol logo SVG component. Renders the official MCP icon
 * with customizable size via class and style props.
 */
export { default as McpLogo } from './McpLogo.svelte';

/**
 *
 * SERVER CARD
 *
 * Components for displaying individual MCP server status and controls.
 * McpServerCard is the main component, with sub-components for specific sections.
 *
 */

/**
 * **McpServerCard** - Individual server display card
 *
 * Main component for displaying a single MCP server with all its details.
 * Manages edit mode, delete confirmation, and health check actions.
 *
 * **Architecture:**
 * - Composes header, tools list, logs, and actions sub-components
 * - Manages local edit/delete state
 * - Reads health state from mcpStore
 * - Triggers health checks via mcpStore
 *
 * **Features:**
 * - Server header with favicon, name, version, and toggle
 * - Capabilities badges display
 * - Tools list with descriptions
 * - Connection logs viewer
 * - Edit form for URL and headers
 * - Delete confirmation dialog
 * - Skeleton loading states
 */
export { default as McpServerCard } from './McpServerCard/McpServerCard.svelte';

/** Server card header with favicon, name, version badge, and enable toggle. */
export { default as McpServerCardHeader } from './McpServerCard/McpServerCardHeader.svelte';

/** Action buttons row: edit, refresh, delete. */
export { default as McpServerCardActions } from './McpServerCard/McpServerCardActions.svelte';

/** Collapsible tools list showing available server tools with descriptions. */
export { default as McpServerCardToolsList } from './McpServerCard/McpServerCardToolsList.svelte';

/** Inline edit form for server URL and custom headers. */
export { default as McpServerCardEditForm } from './McpServerCard/McpServerCardEditForm.svelte';

/** Delete confirmation dialog with server name display. */
export { default as McpServerCardDeleteDialog } from './McpServerCard/McpServerCardDeleteDialog.svelte';

/**
 * **McpServerInfo** - Server instructions display
 *
 * Collapsible panel showing server-provided instructions.
 * Displays guidance text from the MCP server for users.
 */
export { default as McpServerInfo } from './McpServerInfo.svelte';
