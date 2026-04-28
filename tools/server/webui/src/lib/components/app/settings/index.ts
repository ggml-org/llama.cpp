/**
 * Full chat settings page layout with sidebar, mobile header, and content area.
 * Manages local configuration state, section navigation, and context setup.
 * Accepts an optional `initialSection` prop to override the URL-based section resolution.
 */
export { default as SettingsChat } from './SettingsChat.svelte';

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
 * Provides UI for importing and exporting chat settings configurations.
 */
export { default as SettingsImportExport } from './SettingsImportExport.svelte';

/**
 * MCP Servers configuration panel.
 * Provides UI for managing Model Context Protocol (MCP) server connections.
 */
export { default as SettingsMcpServers } from './SettingsMcpServers.svelte';
