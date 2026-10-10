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
export { default as ModelCapabilities } from './ModelCapabilities.svelte';
export { default as ModelContext } from './ModelContext.svelte';
export { default as ModelId } from './ModelId.svelte';
export { default as ModelAvatar } from './ModelAvatar.svelte';
export { default as ModelDownloadProgressBar } from './ModelDownloadProgressBar.svelte';
export { default as ModelLoadControl } from './ModelLoadControl.svelte';
export { default as ModelOrgAvatar } from './ModelOrgAvatar.svelte';
export { default as ModelRowActions } from './ModelRowActions.svelte';
export { default as ModelsSection } from './ModelsSection.svelte';

/** Capability and modality icon row, shared by every model-id surface. */
export { default as ModelCapabilityIcons } from './ModelCapabilityIcons.svelte';

/** Draft sidecar badges, rendered as `+ [KIND] [QUANT]` per sidecar. */
export { default as ModelDraftSidecars } from './ModelDraftSidecars.svelte';
export * from './ModelsSelector';
