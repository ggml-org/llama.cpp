import { ServerModelStatus } from '$lib/enums';
import { apiFetch, apiPost } from '$lib/utils';

export class ModelsService {
	/**
	 *
	 *
	 * Listing
	 *
	 *
	 */

	/**
	 * Fetch list of models from OpenAI-compatible endpoint.
	 * Works in both MODEL and ROUTER modes.
	 *
	 * @returns List of available models with basic metadata
	 */
	static async list(): Promise<ApiModelListResponse> {
		return apiFetch<ApiModelListResponse>('/v1/models');
	}

	/**
	 * Fetch list of all models with detailed metadata (ROUTER mode).
	 * Returns models with load status, paths, and other metadata
	 * beyond what the OpenAI-compatible endpoint provides.
	 *
	 * @returns List of models with detailed status and configuration info
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
	 * Load a model (ROUTER mode only).
	 * Sends POST request to `/models/load`. Note: the endpoint returns success
	 * before loading completes — use polling to await actual load status.
	 *
	 * @param modelId - Model identifier to load
	 * @param extraArgs - Optional additional arguments to pass to the model instance
	 * @returns Load response from the server
	 */
	static async load(modelId: string, extraArgs?: string[]): Promise<ApiRouterModelsLoadResponse> {
		const payload: { model: string; extra_args?: string[] } = { model: modelId };
		if (extraArgs && extraArgs.length > 0) {
			payload.extra_args = extraArgs;
		}

		return apiPost<ApiRouterModelsLoadResponse>('/models/load', payload);
	}

	/**
	 * Unload a model (ROUTER mode only).
	 * Sends POST request to `/models/unload`. Note: the endpoint returns success
	 * before unloading completes — use polling to await actual unload status.
	 *
	 * @param modelId - Model identifier to unload
	 * @returns Unload response from the server
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
	 * Check if a model is loaded based on its metadata.
	 *
	 * @param model - Model data entry from the API response
	 * @returns True if the model status is LOADED
	 */
	static isModelLoaded(model: ApiModelDataEntry): boolean {
		return model.status.value === ServerModelStatus.LOADED;
	}

	/**
	 * Check if a model is currently loading.
	 *
	 * @param model - Model data entry from the API response
	 * @returns True if the model status is LOADING
	 */
	static isModelLoading(model: ApiModelDataEntry): boolean {
		return model.status.value === ServerModelStatus.LOADING;
	}
}
