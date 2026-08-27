/**
 * ModelsService - Stateless model management API layer
 *
 * Wraps the /models endpoints (list, load, unload) and the /models/sse
 * status feed in MODEL and ROUTER modes. No reactive state; consumed by
 * modelsStore and its status manager.
 */

import { base } from '$app/paths';
import { API_MODELS, type DraftVariant, MODEL_ID } from '$lib/constants';
import { ServerModelStatus } from '$lib/enums';
import type { ParsedModelId } from '$lib/types/models';
import {
	apiDelete,
	apiFetch,
	apiPost,
	extractSseDataPayload,
	normalizeModelName,
	splitSseRecords
} from '$lib/utils';
import { getAuthHeaders } from '$lib/utils/api-headers';

/** Building block for the `<repo>:<tag>` string consumed by POST /models. */
export interface GgufVariantTagInput {
	quant: string;
	variant: DraftVariant | null;
}

export class ModelsService {
	private static readonly SSE_RECONNECT_MS = 1000;

	/**
	 * Build the `<repo>:<tag>` string expected by POST /models from a parsed
	 * filename quant + optional draft variant. Used by the model-hub download
	 * dialog so callers don't have to know about the suffix convention.
	 *
	 * @param repoId - HuggingFace repo id (e.g. `ggml-org/gemma-3-4b-it-GGUF`)
	 * @param quant - Quantization token, may include a draft variant
	 * @returns Repo id possibly suffixed with `:tag`
	 */
	static buildDownloadTag(repoId: string, quant: GgufVariantTagInput | null): string {
		if (!quant) return repoId;

		const tag = quant.variant ? `${quant.quant}-${quant.variant.toUpperCase()}` : quant.quant;

		return `${repoId}:${tag}`;
	}

	/**
	 *
	 *
	 * Download
	 *
	 *
	 */

	/**
	 * Cancel an in-flight download or remove a previously downloaded/failed
	 * entry from the server's model cache (ROUTER mode only).
	 *
	 * Sends DELETE `/models?model=<hfRepoWithTag>`:
	 * - while a download is running, the child subprocess is asked to exit
	 *   and any partial `.tmp` files are removed;
	 * - once the entry has finished downloading or has failed, the cached
	 *   files are removed from disk.
	 *
	 * @param hfRepoWithTag - HuggingFace repo id in the same `<repo>:<tag>`
	 *                        format returned by `buildDownloadTag`.
	 * @returns Server acknowledgement containing the success flag
	 */
	static async cancelDownload(hfRepoWithTag: string): Promise<ApiRouterModelsDownloadResponse> {
		return apiDelete<ApiRouterModelsDownloadResponse>(API_MODELS.DELETE, {
			model: hfRepoWithTag
		});
	}

	/**
	 * Trigger a model download from HuggingFace (ROUTER mode only).
	 *
	 * Sends a POST request to `/models` as introduced in
	 * ggml-org/llama.cpp#23976. The response returns immediately; the actual
	 * download runs in the background and tracks progress through `/models/sse`.
	 * The server picks the file that matches the supplied tag (when present)
	 * and additionally pulls mmproj / MTP sidecar weights as appropriate for
	 * the model.
	 *
	 * @param hfRepoWithTag - HuggingFace repo id, optionally suffixed with
	 *                        `:<tag>` (e.g. `ggml-org/gemma-3-4b-it-GGUF:Q4_K_M`
	 *                        or `:IQ1_M-MTP` for an embedded-draft GGUF).
	 * @returns Server acknowledgement containing the success flag
	 */
	static async downloadModel(hfRepoWithTag: string): Promise<ApiRouterModelsDownloadResponse> {
		const payload: ApiRouterModelsDownloadRequest = { model: hfRepoWithTag };

		return apiPost<ApiRouterModelsDownloadResponse>(API_MODELS.DOWNLOAD, payload);
	}

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
	 *
	 *
	 * Load/Unload
	 *
	 *
	 */

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
			activatedParams: null,
			modelName: null,
			orgName: null,
			params: null,
			quantization: null,
			raw: modelId,
			tags: [],
			variant: null
		};

		// strip directory path and weight extension so a bare `-m /path/file.gguf`
		// parses like a clean repo id; the HF `org/model` form is preserved
		let source = normalizeModelName(modelId).replace(MODEL_ID.WEIGHT_EXTENSION_RE, '');

		// 0. Detect sidecar variant prefix (mtp-, dflash-, mmproj-) before any other
		//    splitting so the inner id parses cleanly.
		const prefixVariantMatch = source.match(MODEL_ID.DRAFT_VARIANT_PREFIX_RE);

		if (prefixVariantMatch) {
			result.variant = prefixVariantMatch[1].toLowerCase() as DraftVariant;
			source = prefixVariantMatch[2];
		} else {
			// 0b. Detect `-<variant>` suffix (`-mtp`, `-dflash`, `-dspark`, `-eagle3`).
			//     Only strip it when the segment preceding it looks like a real quant
			//     token, so a model literally named `MyModel-mtp` is not mistaken for a
			//     draft one.
			const suffixMatch = source.match(MODEL_ID.DRAFT_VARIANT_SUFFIX_RE);

			if (suffixMatch) {
				const candidate = suffixMatch[1];
				const headSeg = candidate.split(MODEL_ID.SEGMENT_SEPARATOR).pop();

				if (headSeg && MODEL_ID.QUANTIZATION_SEGMENT_RE.test(headSeg)) {
					result.variant = suffixMatch[2].toLowerCase() as DraftVariant;
					source = candidate;
				}
			}
		}

		// 1. Extract colon-separated quantization (e.g. `model:Q4_K_M`)
		const colonIdx = source.indexOf(MODEL_ID.QUANTIZATION_SEPARATOR);

		let modelPath: string;

		if (colonIdx !== MODEL_ID.NOT_FOUND) {
			result.quantization = source.slice(colonIdx + 1) || null;
			modelPath = source.slice(0, colonIdx);
		} else {
			modelPath = source;
		}

		// 2. Extract org name (e.g. `org/model` -> org = "org")
		const slashIdx = modelPath.indexOf(MODEL_ID.ORG_SEPARATOR);

		let modelStr: string;

		if (slashIdx !== MODEL_ID.NOT_FOUND) {
			result.orgName = modelPath.slice(0, slashIdx);
			modelStr = modelPath.slice(slashIdx + 1);
		} else {
			modelStr = modelPath;
		}

		// 3. Handle dot-separated quantization (e.g. `model-name.Q4_K_M`)
		const dotIdx = modelStr.lastIndexOf('.');

		if (dotIdx !== MODEL_ID.NOT_FOUND && !result.quantization) {
			const afterDot = modelStr.slice(dotIdx + 1);

			if (MODEL_ID.QUANTIZATION_SEGMENT_RE.test(afterDot)) {
				result.quantization = afterDot;
				modelStr = modelStr.slice(0, dotIdx);
			}
		}

		const segments = modelStr.split(MODEL_ID.SEGMENT_SEPARATOR);

		// 4. Detect trailing quantization from dash-separated segments
		//    Handle UD-prefixed quantization (e.g. `UD-Q8_K_XL`) and
		//    standalone quantization (e.g. `Q4_K_M`, `BF16`, `F16`, `MXFP4`)
		if (!result.quantization && segments.length > 1) {
			const last = segments[segments.length - 1];
			const secondLast = segments.length > 2 ? segments[segments.length - 2] : null;

			if (MODEL_ID.QUANTIZATION_SEGMENT_RE.test(last)) {
				if (secondLast && MODEL_ID.CUSTOM_QUANTIZATION_PREFIX_RE.test(secondLast)) {
					result.quantization = `${secondLast}-${last}`;
					segments.splice(segments.length - 2, 2);
				} else {
					result.quantization = last;
					segments.pop();
				}
			}
		}

		// 5. Find params and activated params
		let paramsIdx = MODEL_ID.NOT_FOUND;
		let activatedParamsIdx = MODEL_ID.NOT_FOUND;

		for (let i = 0; i < segments.length; i++) {
			const seg = segments[i];

			if (paramsIdx === MODEL_ID.NOT_FOUND && MODEL_ID.PARAMS_RE.test(seg)) {
				paramsIdx = i;
				result.params = seg.toUpperCase();
			} else if (paramsIdx !== MODEL_ID.NOT_FOUND && MODEL_ID.ACTIVATED_PARAMS_RE.test(seg)) {
				activatedParamsIdx = i;
				result.activatedParams = seg.toUpperCase();
			}
		}

		// 6. Model name = segments before params; tags = remaining segments after params
		const pivotIdx = paramsIdx !== MODEL_ID.NOT_FOUND ? paramsIdx : segments.length;
		const modelSegments = segments.slice(0, pivotIdx);

		// strip trailing container-format segments (e.g. GGUF) from the model name
		while (
			modelSegments.length > 0 &&
			MODEL_ID.IGNORED_SEGMENTS.has(modelSegments[modelSegments.length - 1].toUpperCase())
		) {
			modelSegments.pop();
		}

		result.modelName = modelSegments.join(MODEL_ID.SEGMENT_SEPARATOR) || null;

		if (paramsIdx !== MODEL_ID.NOT_FOUND) {
			result.tags = segments.slice(paramsIdx + 1).filter((_, relIdx) => {
				const absIdx = paramsIdx + 1 + relIdx;

				if (absIdx === activatedParamsIdx) return false;

				return !MODEL_ID.IGNORED_SEGMENTS.has(segments[absIdx].toUpperCase());
			});
		}

		return result;
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
	 * Read the /models/sse feed and invoke onEvent for each parsed envelope.
	 * Reconnects on network drops until the signal aborts. Splits the byte
	 * stream into SSE records on the blank line boundary; the payload rides in
	 * the data lines as a JSON envelope with its own model, event and data fields.
	 */
	static async watchModelEvents(
		signal: AbortSignal,
		onEvent: (event: ApiModelsSseEvent) => void
	): Promise<void> {
		const decoder = new TextDecoder();

		while (!signal.aborted) {
			try {
				const response = await fetch(`${base}${API_MODELS.SSE}`, {
					headers: getAuthHeaders(),
					signal
				});

				if (response.ok && response.body) {
					const reader = response.body.getReader();

					let buffer = '';

					while (!signal.aborted) {
						const { done, value } = await reader.read();

						if (done) break;

						buffer += decoder.decode(value, { stream: true });

						const { records, rest } = splitSseRecords(buffer);

						buffer = rest;

						for (const record of records) {
							const event = ModelsService.parseStatusRecord(record);

							if (event) onEvent(event);
						}
					}
				}
			} catch {
				// network drop or abort falls through to the reconnect delay
			}

			if (signal.aborted) return;

			await new Promise((resolve) => setTimeout(resolve, ModelsService.SSE_RECONNECT_MS));
		}
	}

	/**
	 * Parse one SSE record into its JSON envelope, or null when the record
	 * carries no data payload or malformed JSON.
	 */
	private static parseStatusRecord(record: string): ApiModelsSseEvent | null {
		const payload = extractSseDataPayload(record);

		if (payload.length === 0) return null;

		try {
			return JSON.parse(payload) as ApiModelsSseEvent;
		} catch {
			return null;
		}
	}
}
