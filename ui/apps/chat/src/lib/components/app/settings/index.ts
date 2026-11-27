/**
 * Full chat settings page layout with sidebar, mobile header, and content area.
 * Manages local configuration state, section navigation, and context setup.
 * Accepts an optional `initialSection` prop to set the initial active section.
 */
export { default as SettingsChat } from './SettingsChat/SettingsChat.svelte';

/**
 * Desktop sidebar navigation for chat settings.
 * Displays a list of settings sections with icons and titles.
 * Switches sections in-app via `onSectionChange`.
 */
export { default as SettingsChatDesktopSidebar } from './SettingsChatDesktopSidebar.svelte';

/**
 * Mobile header with a horizontally scrollable section picker for chat settings.
 * Shows chevron buttons for scroll navigation and highlights the active section.
 * Switches sections in-app via `onSectionChange`.
 */
export { default as SettingsChatMobileHeader } from './SettingsChatMobileHeader.svelte';

/**
 * Badge indicating parameter source for sampling settings. Shows one of:
 * - **Custom**: User has explicitly set this value (orange badge)
 * - **Server Props**: Using default from `/props` endpoint (blue badge)
 * - **Default**: Using app default, server props unavailable (gray badge)
 * Updates in real-time as user types to show immediate feedback.
 */
export { default as SettingsChatParameterSourceIndicator } from './SettingsChat/SettingsChatParameterSourceIndicator.svelte';

/**
 * Section wrapper for settings panels. Displays a title heading with
 * child content in a structured layout.
 */
export { default as SettingsGroup } from './SettingsGroup.svelte';

/**
 * The Hugging Face account the app is signed in with, and its Sign out. Shows nothing until
 * a share has signed the app in.
 */
export { default as SettingsHuggingFaceAccount } from './SettingsHuggingFaceAccount.svelte';

/**
 * Sticky footer with the save button for settings panel, shown only when
 * there are unsaved changes. Save commits form state to config store.
 */
export { default as SettingsFooter } from './SettingsFooter.svelte';

/**
 * "Reset to default" button with confirmation dialog, shown only when the
 * saved config differs from the defaults.
 */
export { default as SettingsResetButton } from './SettingsResetButton.svelte';

/**
 * Settings Import/Export panel.
 * Provides UI for importing and exporting chat conversations.
 */
export { default as SettingsChatImportExportTab } from './SettingsChat/SettingsChatImportExportTab.svelte';

/**
 * **SettingsChatVoiceTab** - Voice section of the settings
 *
 * The registry's voice fields, with the transcription model picker between them:
 * in router mode it lists the server's models with audio input, on a single-model
 * server it says whether that model can transcribe.
 */
export { default as SettingsChatVoiceTab } from './SettingsChat/SettingsChatVoiceTab.svelte';

/**
 * Section wrapper for import/export sections. Displays a title, description,
 * icon button, and optional summary of recent actions.
 */
export { default as SettingsChatImportExportSection } from './SettingsChat/SettingsChatImportExportSection.svelte';

/**
 * MCP Servers configuration panel.
 * Provides UI for managing Model Context Protocol (MCP) server connections.
 */
export { default as SettingsMcpServers } from './SettingsMcpServers.svelte';

/**
 * Form fields renderer for individual settings. Generates appropriate input
 * components based on field type (text, number, select, checkbox, textarea).
 * Handles validation, help text display, and parameter source indicators.
 */
export { default as SettingsChatFields } from './SettingsChat/SettingsChatFields.svelte';

/**
 * The settings fields as a phone's native inset list: switch, picker and number rows grouped,
 * prompts and single choices as groups of their own. Hands each change on as it is made.
 */
export { default as SettingsChatFieldsInset } from './SettingsChat/SettingsChatFieldsInset.svelte';

/**
 * Phone settings: a full-height sheet whose rows push each section, its advanced settings and the
 * tools onto a stack of pages. Changes apply as they are made.
 */
export { default as SettingsSheet } from './SettingsSheet/SettingsSheet.svelte';

/**
 * **SettingsChatToolsTab** - Tools configuration tab for chat settings
 *
 * Displays available tools grouped by source (server, browser, MCP, custom) with
 * toggles to enable/disable individual tools and tool groups. Shows MCP
 * server favicons and permission management controls.
 */
export { default as SettingsChatToolsTab } from './SettingsChat/SettingsChatToolsTab.svelte';
