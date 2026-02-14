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
 * **ModelsSelector** - Model selection dropdown
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
 * <ModelsSelector
 *   currentModel={conversation.modelId}
 *   onModelChange={(id, name) => updateModel(id)}
 *   useGlobalSelection
 * />
 * ```
 */
export { default as ModelsSelector } from './ModelsSelector.svelte';

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
