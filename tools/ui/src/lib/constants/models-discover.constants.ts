/**
 * Models discover constants.
 *
 * Endpoints and settings for the Models Discover dialog.
 */

/** llama.app model catalog used as the default model list. Online-only source; the discover feature requires an internet connection anyway. */
export const MODELS_DISCOVER_CATALOG_URL = 'https://llama.app/v1/catalog.json';

/**
 * Catalog repos fetched in parallel per batch; small on purpose so the HF API
 * is not hit with the whole catalog at once.
 */
export const MODELS_DISCOVER_CATALOG_BATCH = 4;
