import { ServerModelStatus } from '$lib/enums';
import { apiFetch, apiPost } from '$lib/utils';

/**
 * ModelsService - Stateless service for model management API communication
 *
 * This service handles communication with model-related endpoints:
 * - `/v1/models` - OpenAI-compatible model list (MODEL + ROUTER mode)
 * - `/models/load`, `/models/unload` - Router-specific model management (ROUTER mode only)
 *
 * **Responsibilities:**
 * - List available models
 * - Load/unload models (ROUTER mode)
 * - Check model status (ROUTER mode)
 *
 * **Used by:**
 * - modelsStore: Primary consumer for model state management
 */
export class ModelsService {
	/**
	 *
	 *
	 * Listing
	 *
	 *
	 */

	/**
	 * Fetch list of models from OpenAI-compatible endpoint
	 * Works in both MODEL and ROUTER modes
	 */
	static async list(): Promise<ApiModelListResponse> {
		return apiFetch<ApiModelListResponse>('/v1/models');
	}

	/**
	 * Fetch list of all models with detailed metadata (ROUTER mode)
	 * Returns models with load status, paths, and other metadata
	 */
	static async listRouter(): Promise<ApiRouterModelsListResponse> {
		return apiFetch<ApiRouterModelsListResponse>('/v1/models');
	}

	/**
	 *
	 *
	 * Load/Unload
	 *
	 *
	 */

	/**
	 * Load a model (ROUTER mode)
	 * POST /models/load
	 * @param modelId - Model identifier to load
	 * @param extraArgs - Optional additional arguments to pass to the model instance
	 */
	static async load(modelId: string, extraArgs?: string[]): Promise<ApiRouterModelsLoadResponse> {
		const payload: { model: string; extra_args?: string[] } = { model: modelId };
		if (extraArgs && extraArgs.length > 0) {
			payload.extra_args = extraArgs;
		}

		return apiPost<ApiRouterModelsLoadResponse>('/models/load', payload);
	}

	/**
	 * Unload a model (ROUTER mode)
	 * POST /models/unload
	 * @param modelId - Model identifier to unload
	 */
	static async unload(modelId: string): Promise<ApiRouterModelsUnloadResponse> {
		return apiPost<ApiRouterModelsUnloadResponse>('/models/unload', { model: modelId });
	}

	/**
	 *
	 *
	 * Status
	 *
	 *
	 */

	/**
	 * Check if a model is loaded based on its metadata
	 */
	static isModelLoaded(model: ApiModelDataEntry): boolean {
		return model.status.value === ServerModelStatus.LOADED;
	}

	/**
	 * Check if a model is currently loading
	 */
	static isModelLoading(model: ApiModelDataEntry): boolean {
		return model.status.value === ServerModelStatus.LOADING;
	}
}
