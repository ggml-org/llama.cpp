/**
 *
 * MODELS
 *
 * Components for model selection and display. Supports two server modes:
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
 * **ModelsSelectorOption** - Single model option row
 *
 * Renders a single model option with selection state, favorite toggle,
 * load/unload actions, status indicators, and an info button.
 * Used inside ModelsSelectorList or directly in custom render snippets.
 */
export { default as ModelsSelectorOption } from './ModelsSelectorOption.svelte';

/**
 * **ModelsSelectorSheet** - Mobile model selection sheet
 *
 * Bottom sheet variant of ModelsSelectorDropdown optimized for touch interaction
 * on mobile devices. Same functionality as ModelsSelectorDropdown but uses Sheet UI
 * instead of DropdownMenu.
 */
export { default as ModelsSelectorSheet } from './ModelsSelectorSheet.svelte';

/**
 * **ModelBadge** - Model name display badge
 *
 * Compact badge showing current model name with package icon.
 * Only visible in single model mode. Supports tooltip and copy functionality.
 *
 * **Architecture:**
 * - Reads model name from modelsStore or prop
 * - Checks server mode from serverStore
 * - Uses BadgeInfo for consistent styling
 *
 * **Features:**
 * - Optional copy to clipboard button
 * - Optional tooltip with model details
 * - Click handler for model info dialog
 * - Only renders in model mode (not router)
 *
 * @example
 * ```svelte
 * <ModelBadge
 *   onclick={() => showModelInfo = true}
 *   showTooltip
 *   showCopyIcon
 * />
 * ```
 */
export { default as ModelBadge } from './ModelBadge.svelte';

/**
 * **ModelId** - Parsed model identifier display
 *
 * Displays a model ID with optional org name, parameter badges, quantization,
 * aliases, and tags. Supports raw mode to show the unprocessed model name.
 * Respects the user's `showRawModelNames` setting.
 */
export { default as ModelId } from './ModelId.svelte';
