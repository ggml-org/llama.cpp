/**
 * Full chat settings page layout with sidebar, mobile header, and content area.
 * Manages local configuration state, section navigation, and context setup.
 * Accepts an optional `initialSection` prop to override the URL-based section resolution.
 */
export { default as SettingsChat } from './SettingsChat/SettingsChat.svelte';

/**
 * Desktop sidebar navigation for chat settings.
 * Displays a list of settings sections with icons and titles.
 * Supports both hash-link navigation (via `getHref`) and in-app section switching (via `onSectionChange`).
 */
export { default as SettingsChatDesktopSidebar } from './SettingsChatDesktopSidebar.svelte';

/**
 * Mobile header with a horizontally scrollable section picker for chat settings.
 * Shows chevron buttons for scroll navigation and highlights the active section.
 * Supports both hash-link navigation (via `getHref`) and in-app section switching (via `onSectionChange`).
 */
export { default as SettingsChatMobileHeader } from './SettingsChatMobileHeader.svelte';

/**
 * Settings Import/Export panel.
 * Provides UI for importing and exporting chat conversations.
 */
export { default as SettingsChatImportExportTab } from './SettingsChat/SettingsChatImportExportTab.svelte';

/**
 * MCP Servers configuration panel.
 * Provides UI for managing Model Context Protocol (MCP) server connections.
 */
export { default as SettingsMcpServers } from './SettingsMcpServers.svelte';

/**
 * Footer with save/cancel buttons for settings panel. Positioned at bottom
 * of settings dialog. Save button commits form state to config store,
 * cancel button triggers reset and close.
 */
export { default as SettingsChatFooter } from './SettingsChat/SettingsChatFooter.svelte';

/**
 * Form fields renderer for individual settings. Generates appropriate input
 * components based on field type (text, number, select, checkbox, textarea).
 * Handles validation, help text display, and parameter source indicators.
 */
export { default as SettingsChatFields } from './SettingsChat/SettingsChatFields.svelte';

/**
 * Badge indicating parameter source for sampling settings. Shows one of:
 * - **Custom**: User has explicitly set this value (orange badge)
 * - **Server Props**: Using default from `/props` endpoint (blue badge)
 * - **Default**: Using app default, server props unavailable (gray badge)
 * Updates in real-time as user types to show immediate feedback.
 */
export { default as SettingsChatParameterSourceIndicator } from './SettingsChat/SettingsChatParameterSourceIndicator.svelte';

/**
 * **SettingsChatToolsTab** - Tools configuration tab for chat settings
 *
 * Displays available tools grouped by source (built-in, MCP, custom) with
 * toggles to enable/disable individual tools and tool groups. Shows MCP
 * server favicons and permission management controls.
 */
export { default as SettingsChatToolsTab } from './SettingsChat/SettingsChatToolsTab.svelte';
