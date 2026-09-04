/**
 *
 * MODELS SELECTOR
 *
 * Model selection UI for the chat form: a desktop dropdown and a mobile sheet,
 * each backed by a shared grouped options list, option rows, in-flight download
 * rows and an inline reasoning-effort picker. Supports two server modes:
 * - **Single model mode**: Server runs with one model, selector shows model info
 * - **Router mode**: Server runs with multiple models, selector enables switching
 *
 * Integrates with modelsStore for model data and serverStore for mode detection.
 *
 */

/**
 * **ModelsSelectorDropdown** - Model selection dropdown (desktop)
 *
 * Dropdown for selecting AI models with status indicators,
 * search, and model information display. Adapts UI based on server mode.
 *
 * **Architecture:**
 * - Uses DropdownMenuSearchable for model list
 * - Integrates with modelsStore for model options and selection
 * - Detects router vs single mode from serverStore
 * - Opens DialogModelInformation for model details
 *
 * **Features:**
 * - Searchable model list with keyboard navigation
 * - Model status indicators (loading/ready/error/updating)
 * - Model capabilities badges (vision, tools, etc.)
 * - Current/active model highlighting
 * - Model information dialog on info button click
 * - Router mode: shows all available models with status
 * - Single mode: shows current model name only
 * - Loading/updating skeleton states
 * - Global selection support for form integration
 *
 * @example
 * ```svelte
 * <ModelsSelectorDropdown
 *   currentModel={conversation.modelId}
 *   onModelChange={(id, name) => updateModel(id)}
 *   useGlobalSelection
 * />
 * ```
 */
export { default as ModelsSelectorDropdown } from './ModelsSelectorDropdown.svelte';

/**
 * **ModelsSelectorList** - Grouped model options list
 *
 * Renders grouped model options (loaded, favorites, available) with section
 * headers and org subgroups. Shared between ModelsSelectorDropdown and ModelsSelectorSheet
 * to avoid template duplication.
 *
 * Accepts an optional `renderOption` snippet to customize how each option is
 * rendered (e.g., to add keyboard navigation or highlighting).
 */
export { default as ModelsSelectorList } from './ModelsSelectorList.svelte';

/**
 * **ModelsSelectorDownloadItem** - Single in-flight download row
 *
 * One "Download in progress" row for the models selector: live progress bar,
 * pause / resume on click and a hover-revealed cancel, mirroring the discover
 * quant chips.
 */
export { default as ModelsSelectorDownloadItem } from './ModelsSelectorDownloadItem.svelte';

/**
 * **ModelsSelectorOption** - Single model option row
 *
 * Renders a single model option with selection state, favorite toggle,
 * load/unload actions, status indicators, and an info button.
 * Used inside ModelsSelectorList or directly in custom render snippets.
 */
export { default as ModelsSelectorOption } from './ModelsSelectorOption.svelte';

/**
 * **ModelsSelectorReasoningPanel** - Inline reasoning effort picker
 *
 * Collapsible row that expands in place to reveal the reasoning effort levels.
 * Used in the sticky footer of ModelsSelectorDropdown, where a flyout submenu
 * would float over the model list it belongs to.
 */
export { default as ModelsSelectorReasoningPanel } from './ModelsSelectorReasoningPanel.svelte';

/**
 * **ModelsSelectorSheet** - Mobile model selection sheet
 *
 * Bottom sheet variant of ModelsSelectorDropdown optimized for touch interaction
 * on mobile devices. Same functionality as ModelsSelectorDropdown but uses Sheet UI
 * instead of DropdownMenu.
 */
export { default as ModelsSelectorSheet } from './ModelsSelectorSheet.svelte';
