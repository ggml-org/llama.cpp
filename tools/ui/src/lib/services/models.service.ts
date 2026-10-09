/**
 * ModelsService - Stateless model management API layer
 *
 * Wraps the /models endpoints (list, load, unload) and the /models/sse
 * status feed in MODEL and ROUTER modes. No reactive state; consumed by
 * modelsStore and its status manager.
 */

import {
	API_MODELS,
	LOCAL_BACKEND_ID,
	MODEL_ID,
	type ModelSidecar,
	SIDECAR_TOKENS
} from '$lib/constants';
import { ServerModelStatus } from '$lib/enums';
import type { ModelSidecarFile, ParsedModelId } from '$lib/types/models';
import {
	apiDelete,
	apiFetch,
	apiModelsUrl,
	apiPost,
	apiUrl,
	extractSseDataPayload,
	normalizeModelName,
	sidecarFromFileToken,
	sidecarFromTag,
	splitSseRecords
} from '$lib/utils';
import { getAuthHeaders } from '$lib/utils/api-headers';
import { isAuxSidecar } from '$lib/utils/sidecars';

/** Sidecar token a file name carries, for the forms parsing an id as a model misses. */
function sidecarTokenInFilename(modelId: string): ModelSidecar | null {
	// the token can sit after a colon (`org/model:mtp`), a dash or an underscore
	const name = modelId.toLowerCase();
	const match = SIDECAR_TOKENS.find((token) =>
		new RegExp(`(^|[-_:])${token}([-_.:]|$)`).test(name)
	);

	return (match as ModelSidecar | undefined) ?? null;
}

/** Parameter label a parsed id reports, e.g. `35B-A3B`. */
function paramsLabel(parsed: ParsedModelId): string | null {
	if (!parsed.params) return null;

	return `${parsed.params}${parsed.activatedParams ? `-${parsed.activatedParams}` : ''}`;
}

export class ModelsService {
	private static readonly SSE_RECONNECT_MS = 1000;

	/**
	 * Build the `<repo>:<tag>` string POST /models expects, so callers don't need
	 * to know the tag conventions.
	 */
	static buildDownloadTag(
		repoId: string,
		quant: string | null,
		sidecar: ModelSidecar | null
	): string {
		if (!quant && !sidecar) return repoId;

		if (!quant) return `${repoId}:${sidecar}`;

		const tag = sidecar ? `${quant}-${sidecar}` : quant;

		return `${repoId}:${tag}`;
	}

	/**
	 * Cancel an in-flight download, or remove a downloaded/failed entry from the
	 * model cache (ROUTER mode only): DELETE /models?model=<repo:tag>.
	 */
	static async cancelDownload(hfRepoWithTag: string): Promise<ApiModelsDownloadResponse> {
		return apiDelete<ApiModelsDownloadResponse>(API_MODELS.DELETE, {
			model: hfRepoWithTag
		});
	}

	/**
	 * Start a model download from HuggingFace (ROUTER mode only). The response
	 * returns immediately; progress arrives over /models/sse. The server picks
	 * the file matching the tag and also pulls the model's mmproj/draft sidecars.
	 */
	static async downloadModel(hfRepoWithTag: string): Promise<ApiModelsDownloadResponse> {
		const payload: ApiModelsDownloadRequest = { model: hfRepoWithTag };

		return apiPost<ApiModelsDownloadResponse>(API_MODELS.DOWNLOAD, payload);
	}

	/**
	 * True when a router entry id is a sidecar-only entry, e.g. `org/model:Q4_0-mtp`
	 * or `org/model:mmproj`. Such entries mark a downloaded sidecar file, not a
	 * loadable model, so the selector skips them.
	 */
	/**
	 * Draft sidecars a listing carries as their own entries, keyed by the repo they
	 * belong to. The router lists a downloaded sidecar as a model of its own, so this
	 * keeps the pairing that the model list itself is filtered to drop.
	 */
	static draftSidecarsByRepo(response: ApiModelsListResponse): Record<string, ModelSidecarFile[]> {
		const byRepo: Record<string, ModelSidecarFile[]> = {};

		for (const entry of response.data ?? []) {
			const parsed = ModelsService.parseModelId(entry.id);
			const sidecar = parsed.sidecar ?? sidecarTokenInFilename(entry.id);

			if (!sidecar || isAuxSidecar(sidecar)) continue;

			// the repo is the id with its quant tag and sidecar token taken off; naming
			// parts of the id are not touched, a model name may carry `-4b` for instance
			const model = entry.id
				.split(MODEL_ID.QUANTIZATION_SEPARATOR)[0]
				.replace(MODEL_ID.WEIGHT_EXTENSION_REGEX, '')
				.replace(new RegExp(`[-_ ]?${sidecar}([-_ ]?draft)?$`, 'i'), '');

			if (!model) continue;

			const files = (byRepo[model] ??= []);

			if (files.some((file) => file.kind === sidecar)) continue;

			files.push({
				id: entry.id,
				kind: sidecar,
				model,
				params: paramsLabel(parsed),
				quant: parsed.quantization
			});
		}

		return byRepo;
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

	static isModelLoading(model: ApiModelDataEntry): boolean {
		return model.status.value === ServerModelStatus.LOADING;
	}

	static isSidecarEntry(modelId: string): boolean {
		const idx = modelId.indexOf(MODEL_ID.QUANTIZATION_SEPARATOR);

		if (idx !== MODEL_ID.NOT_FOUND && sidecarFromTag(modelId.slice(idx + 1))) return true;

		// the router also lists projector and draft files by filename, e.g.
		// `org/model-mmproj-F16.gguf` or `mmproj-model.gguf`
		const name = modelId.split('/').pop() ?? modelId;

		return (
			MODEL_ID.SIDECAR_INFIX_REGEX.test(name) ||
			MODEL_ID.SIDECAR_PREFIX_REGEX.test(name) ||
			MODEL_ID.SIDECAR_SUFFIX_REGEX.test(name)
		);
	}

	/**
	 * Fetch list of models from OpenAI-compatible endpoint.
	 * Works in both MODEL and ROUTER modes.
	 *
	 * @returns List of available models with basic metadata
	 */
	static async list(): Promise<ApiModelsListResponse> {
		return apiFetch<ApiModelsListResponse>(apiModelsUrl());
	}

	/**
	 * Load a model (ROUTER mode only).
	 * Sends POST request to `/models/load`. Note: the endpoint returns success
	 * before loading completes — use polling to await actual load status.
	 *
	 * @param modelId - Model identifier to load
	 * @param extraArgs - Optional additional arguments to pass to the model instance
	 * @param backendId - Backend serving the model; the active one when omitted
	 * @returns Load response from the server
	 */
	static async load(
		modelId: string,
		extraArgs?: string[],
		backendId?: string
	): Promise<ApiModelsLoadResponse> {
		const payload: { model: string; extra_args?: string[] } = { model: modelId };

		if (extraArgs && extraArgs.length > 0) {
			payload.extra_args = extraArgs;
		}

		return apiPost<ApiModelsLoadResponse>(API_MODELS.LOAD, payload, { backendId });
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
			sidecar: null,
			tags: []
		};

		// strip directory path and weight extension so a bare `-m /path/file.gguf`
		// parses like a clean repo id; the HF `org/model` form is preserved
		let source = normalizeModelName(modelId).replace(MODEL_ID.WEIGHT_EXTENSION_REGEX, '');

		// 0. Detect sidecar prefix (mtp-, dflash-, mmproj-) before any other
		//    splitting so the inner id parses cleanly.
		const prefixMatch = source.match(MODEL_ID.SIDECAR_PREFIX_REGEX);

		if (prefixMatch) {
			result.sidecar = sidecarFromFileToken(prefixMatch[1].toLowerCase());
			source = prefixMatch[2];

			// a sidecar filename's remainder may be just the quant token,
			// e.g. `mtp-Q4_0.gguf` or `mmproj-F16.gguf`
			if (MODEL_ID.QUANTIZATION_SEGMENT_REGEX.test(source)) {
				result.quantization = source.toUpperCase();
				source = '';
			}
		} else {
			// 0b. Detect `-<type>` suffix (`-mtp`, `-dflash`, `-dspark`, `-eagle3`).
			//     Only strip it when the segment preceding it looks like a real quant
			//     token, so a model literally named `MyModel-mtp` is not mistaken for a
			//     draft one.
			const suffixMatch = source.match(MODEL_ID.SIDECAR_SUFFIX_REGEX);

			if (suffixMatch) {
				const candidate = suffixMatch[1];
				const headSeg = candidate.split(MODEL_ID.SEGMENT_SEPARATOR).pop();

				if (headSeg && MODEL_ID.QUANTIZATION_SEGMENT_REGEX.test(headSeg)) {
					result.sidecar = sidecarFromFileToken(suffixMatch[2].toLowerCase());
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

			if (MODEL_ID.QUANTIZATION_SEGMENT_REGEX.test(afterDot)) {
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

			if (MODEL_ID.QUANTIZATION_SEGMENT_REGEX.test(last)) {
				if (secondLast && MODEL_ID.CUSTOM_QUANTIZATION_PREFIX_REGEX.test(secondLast)) {
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

			if (paramsIdx === MODEL_ID.NOT_FOUND && MODEL_ID.PARAMS_REGEX.test(seg)) {
				paramsIdx = i;
				result.params = seg.toUpperCase();
			} else if (paramsIdx !== MODEL_ID.NOT_FOUND && MODEL_ID.ACTIVATED_PARAMS_REGEX.test(seg)) {
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
	 * @param backendId - Backend serving the model; the active one when omitted
	 * @returns Unload response from the server
	 */
	static async unload(modelId: string, backendId?: string): Promise<ApiModelsUnloadResponse> {
		return apiPost<ApiModelsUnloadResponse>(API_MODELS.UNLOAD, { model: modelId }, { backendId });
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
				// the status feed only exists on the local llama.cpp server; pin the
				// request so an active external backend cannot redirect it
				const response = await fetch(apiUrl(API_MODELS.SSE, LOCAL_BACKEND_ID), {
					headers: getAuthHeaders(LOCAL_BACKEND_ID),
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
