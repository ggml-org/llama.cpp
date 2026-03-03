import { ServerModelStatus } from '$lib/enums';
import { apiFetch, apiPost } from '$lib/utils';
import type { ParsedModelId } from '$lib/types/models';
import {
	MODEL_FORMAT_SEGMENT_RE,
	MODEL_PARAMS_RE,
	MODEL_ACTIVATED_PARAMS_RE,
	MODEL_ID_NOT_FOUND,
	MODEL_ID_ORG_SEPARATOR,
	MODEL_ID_SEGMENT_SEPARATOR,
	MODEL_ID_QUANTIZATION_SEPARATOR,
	API_MODELS
} from '$lib/constants';

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
		return apiFetch<ApiModelListResponse>(API_MODELS.LIST);
	}

	/**
	 * Fetch list of all models with detailed metadata (ROUTER mode).
	 * Returns models with load status, paths, and other metadata
	 * beyond what the OpenAI-compatible endpoint provides.
	 *
	 * @returns List of models with detailed status and configuration info
	 */
	static async listRouter(): Promise<ApiRouterModelsListResponse> {
		return apiFetch<ApiRouterModelsListResponse>(API_MODELS.LIST);
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

		return apiPost<ApiRouterModelsLoadResponse>(API_MODELS.LOAD, payload);
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
		return apiPost<ApiRouterModelsUnloadResponse>(API_MODELS.UNLOAD, { model: modelId });
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

	/**
	 *
	 *
	 * Parsing
	 *
	 *
	 */

	/**
	 * Parse a model ID string into its structured components.
	 *
	 * Handles the convention:
	 *   `<org>/<ModelName>-<Parameters>(-<ActivatedParameters>)-<Format>:<QuantizationType>`
	 *
	 * @param modelId - Raw model identifier string
	 * @returns Structured {@link ParsedModelId} with all detected fields
	 */
	static parseModelId(modelId: string): ParsedModelId {
		const result: ParsedModelId = {
			raw: modelId,
			orgName: null,
			modelName: null,
			params: null,
			activatedParams: null,
			format: null,
			quantization: null,
			tags: []
		};

		const colonIdx = modelId.indexOf(MODEL_ID_QUANTIZATION_SEPARATOR);
		let modelPath: string;

		if (colonIdx !== MODEL_ID_NOT_FOUND) {
			result.quantization = modelId.slice(colonIdx + 1) || null;
			modelPath = modelId.slice(0, colonIdx);
		} else {
			modelPath = modelId;
		}

		const slashIdx = modelPath.indexOf(MODEL_ID_ORG_SEPARATOR);
		let modelStr: string;

		if (slashIdx !== MODEL_ID_NOT_FOUND) {
			result.orgName = modelPath.slice(0, slashIdx);
			modelStr = modelPath.slice(slashIdx + 1);
		} else {
			modelStr = modelPath;
		}

		const segments = modelStr.split(MODEL_ID_SEGMENT_SEPARATOR);

		if (segments.length > 0 && MODEL_FORMAT_SEGMENT_RE.test(segments[segments.length - 1])) {
			result.format = segments.pop()!;
		}

		const paramsRe = MODEL_PARAMS_RE;
		const activatedParamsRe = MODEL_ACTIVATED_PARAMS_RE;

		let paramsIdx = MODEL_ID_NOT_FOUND;
		let activatedParamsIdx = MODEL_ID_NOT_FOUND;

		for (let i = 0; i < segments.length; i++) {
			const seg = segments[i];
			if (paramsIdx === -1 && paramsRe.test(seg)) {
				paramsIdx = i;
				result.params = seg.toUpperCase();
			} else if (activatedParamsRe.test(seg)) {
				activatedParamsIdx = i;
				result.activatedParams = seg.toUpperCase();
			}
		}

		const pivotIdx = paramsIdx !== MODEL_ID_NOT_FOUND ? paramsIdx : segments.length;

		result.modelName = segments.slice(0, pivotIdx).join(MODEL_ID_SEGMENT_SEPARATOR) || null;

		if (paramsIdx !== MODEL_ID_NOT_FOUND) {
			result.tags = segments
				.slice(paramsIdx + 1)
				.filter((_, relIdx) => paramsIdx + 1 + relIdx !== activatedParamsIdx);
		}

		return result;
	}
}
