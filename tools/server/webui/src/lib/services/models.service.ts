import { ServerModelStatus } from '$lib/enums';
import { apiFetch, apiPost } from '$lib/utils';
import type { ParsedModelId } from '$lib/types/models';
import {
	MODEL_QUANTIZATION_SEGMENT_RE,
	MODEL_CUSTOM_QUANTIZATION_PREFIX_RE,
	MODEL_PARAMS_RE,
	MODEL_ACTIVATED_PARAMS_RE,
	MODEL_IGNORED_SEGMENTS,
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
	 * Handles conventions like:
	 *   `<org>/<ModelName>-<Parameters>(-<ActivatedParameters>)(-<Tags>)(-<Quantization>):<Quantization>`
	 *   `<ModelName>.<Quantization>` (dot-separated quantization, e.g. `model.Q4_K_M`)
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
			quantization: null,
			tags: []
		};

		// 1. Extract colon-separated quantization (e.g. `model:Q4_K_M`)
		const colonIdx = modelId.indexOf(MODEL_ID_QUANTIZATION_SEPARATOR);
		let modelPath: string;

		if (colonIdx !== MODEL_ID_NOT_FOUND) {
			result.quantization = modelId.slice(colonIdx + 1) || null;
			modelPath = modelId.slice(0, colonIdx);
		} else {
			modelPath = modelId;
		}

		// 2. Extract org name (e.g. `org/model` -> org = "org")
		const slashIdx = modelPath.indexOf(MODEL_ID_ORG_SEPARATOR);
		let modelStr: string;

		if (slashIdx !== MODEL_ID_NOT_FOUND) {
			result.orgName = modelPath.slice(0, slashIdx);
			modelStr = modelPath.slice(slashIdx + 1);
		} else {
			modelStr = modelPath;
		}

		// 3. Handle dot-separated quantization (e.g. `model-name.Q4_K_M`)
		const dotIdx = modelStr.lastIndexOf('.');

		if (dotIdx !== MODEL_ID_NOT_FOUND && !result.quantization) {
			const afterDot = modelStr.slice(dotIdx + 1);

			if (MODEL_QUANTIZATION_SEGMENT_RE.test(afterDot)) {
				result.quantization = afterDot;
				modelStr = modelStr.slice(0, dotIdx);
			}
		}

		const segments = modelStr.split(MODEL_ID_SEGMENT_SEPARATOR);

		// 4. Detect trailing quantization from dash-separated segments
		//    Handle UD-prefixed quantization (e.g. `UD-Q8_K_XL`) and
		//    standalone quantization (e.g. `Q4_K_M`, `BF16`, `F16`, `MXFP4`)
		if (!result.quantization && segments.length > 1) {
			const last = segments[segments.length - 1];
			const secondLast = segments.length > 2 ? segments[segments.length - 2] : null;

			if (MODEL_QUANTIZATION_SEGMENT_RE.test(last)) {
				if (secondLast && MODEL_CUSTOM_QUANTIZATION_PREFIX_RE.test(secondLast)) {
					result.quantization = `${secondLast}-${last}`;
					segments.splice(segments.length - 2, 2);
				} else {
					result.quantization = last;
					segments.pop();
				}
			}
		}

		// 5. Find params and activated params
		let paramsIdx = MODEL_ID_NOT_FOUND;
		let activatedParamsIdx = MODEL_ID_NOT_FOUND;

		for (let i = 0; i < segments.length; i++) {
			const seg = segments[i];

			if (paramsIdx === MODEL_ID_NOT_FOUND && MODEL_PARAMS_RE.test(seg)) {
				paramsIdx = i;
				result.params = seg.toUpperCase();
			} else if (paramsIdx !== MODEL_ID_NOT_FOUND && MODEL_ACTIVATED_PARAMS_RE.test(seg)) {
				activatedParamsIdx = i;
				result.activatedParams = seg.toUpperCase();
			}
		}

		// 6. Model name = segments before params; tags = remaining segments after params
		const pivotIdx = paramsIdx !== MODEL_ID_NOT_FOUND ? paramsIdx : segments.length;

		result.modelName = segments.slice(0, pivotIdx).join(MODEL_ID_SEGMENT_SEPARATOR) || null;

		if (paramsIdx !== MODEL_ID_NOT_FOUND) {
			result.tags = segments.slice(paramsIdx + 1).filter((_, relIdx) => {
				const absIdx = paramsIdx + 1 + relIdx;
				if (absIdx === activatedParamsIdx) return false;

				return !MODEL_IGNORED_SEGMENTS.has(segments[absIdx].toUpperCase());
			});
		}

		return result;
	}
}
