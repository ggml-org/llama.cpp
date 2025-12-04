import { base } from '$app/paths';
import { ServerModelStatus } from '$lib/enums';
import { getJsonHeaders } from '$lib/utils';

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
	// ─────────────────────────────────────────────────────────────────────────────
	// Listing
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Fetch list of models from OpenAI-compatible endpoint
	 * Works in both MODEL and ROUTER modes
	 */
	static async list(): Promise<ApiModelListResponse> {
		const response = await fetch(`${base}/v1/models`, {
			headers: getJsonHeaders()
		});

		if (!response.ok) {
			throw new Error(`Failed to fetch model list (status ${response.status})`);
		}

		return response.json() as Promise<ApiModelListResponse>;
	}

	/**
	 * Fetch list of all models with detailed metadata (ROUTER mode)
	 * Returns models with load status, paths, and other metadata
	 */
	static async listRouter(): Promise<ApiRouterModelsListResponse> {
		const response = await fetch(`${base}/v1/models`, {
			headers: getJsonHeaders()
		});

		if (!response.ok) {
			throw new Error(`Failed to fetch router models list (status ${response.status})`);
		}

		return response.json() as Promise<ApiRouterModelsListResponse>;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Load/Unload
	// ─────────────────────────────────────────────────────────────────────────────

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

		const response = await fetch(`${base}/models/load`, {
			method: 'POST',
			headers: getJsonHeaders(),
			body: JSON.stringify(payload)
		});

		if (!response.ok) {
			const errorData = await response.json().catch(() => ({}));
			throw new Error(errorData.error || `Failed to load model (status ${response.status})`);
		}

		return response.json() as Promise<ApiRouterModelsLoadResponse>;
	}

	/**
	 * Unload a model (ROUTER mode)
	 * POST /models/unload
	 * @param modelId - Model identifier to unload
	 */
	static async unload(modelId: string): Promise<ApiRouterModelsUnloadResponse> {
		const response = await fetch(`${base}/models/unload`, {
			method: 'POST',
			headers: getJsonHeaders(),
			body: JSON.stringify({ model: modelId })
		});

		if (!response.ok) {
			const errorData = await response.json().catch(() => ({}));
			throw new Error(errorData.error || `Failed to unload model (status ${response.status})`);
		}

		return response.json() as Promise<ApiRouterModelsUnloadResponse>;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Status
	// ─────────────────────────────────────────────────────────────────────────────

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
