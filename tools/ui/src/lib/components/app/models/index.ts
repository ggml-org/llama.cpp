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
 * The selection UI lives in the ModelsSelector subfolder; the shared model
 * display primitives (id / badge rendering) stay here.
 *
 */

/** * **ModelBadge** - Model name display badge
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

export * from './ModelsSelector';
