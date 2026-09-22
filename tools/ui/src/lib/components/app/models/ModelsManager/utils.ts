import { LOCAL_BACKEND_ID, MODEL_OVERRIDES_LOCALSTORAGE_KEY } from '$lib/constants';
import type { ModelOption } from '$lib/types/models';
import { getBackend } from '$lib/utils/api-base';
import { formatFileSize, formatParameters } from '$lib/utils/formatters';

/** Load parameters a model can override before it is loaded. */
export interface ModelLoadOverride {
	batchSize?: number;
	contextLength?: number;
	cpuThreads?: number;
	flashAttention?: boolean;
	gpuOffload?: number;
	keepInMemory?: boolean;
	speculativeDecoding?: string;
	ubatchSize?: number;
	useMmap?: boolean;
}

/** Sampling parameters. A null value means the server default stays in charge. */
export interface ModelSamplingOverride {
	minP?: number | null;
	repeatPenalty?: number | null;
	temperature?: number | null;
	topK?: number | null;
	topP?: number | null;
}

export interface ModelOverride {
	load?: ModelLoadOverride;
	reasoning?: { budget: string; enabled: boolean };
	sampling?: ModelSamplingOverride;
	stopStrings?: string[];
	structuredOutput?: { enabled: boolean; schema: string };
	systemPrompt?: string;
}

export type ModelOverrideMap = Record<string, ModelOverride>;

/** One collapsible block of the manager's table. */
export interface ModelsTableGroup {
	/** Null for the loaded and favorites groups. */
	backendId: string | null;
	isLocal: boolean;
	items: ModelOption[];
	key: string;
	/** Picks the header icon; providers use their backend logo instead. */
	kind: 'favorites' | 'loaded' | 'local' | 'provider';
	label: string;
}

/** Values the load form falls back to when the server reports nothing. */
export const LOAD_DEFAULTS = {
	batchSize: 2048,
	contextLength: 8192,
	cpuThreads: 13,
	gpuOffload: 42,
	speculativeDecoding: 'off',
	ubatchSize: 512
};

export const SAMPLING_DEFAULTS = {
	minP: 0.05,
	repeatPenalty: 1.1,
	temperature: 1,
	topK: 64,
	topP: 0.95
};

export const SPECULATIVE_OPTIONS = ['off', 'draft-model'];

/** True when the user saved anything for this model. */
export function isCustomized(override?: ModelOverride): boolean {
	return override !== undefined && Object.keys(override).length > 0;
}

export function loadOverrides(): ModelOverrideMap {
	try {
		const raw = localStorage.getItem(MODEL_OVERRIDES_LOCALSTORAGE_KEY);

		if (!raw) return {};

		const parsed = JSON.parse(raw) as unknown;

		return parsed && typeof parsed === 'object' ? (parsed as ModelOverrideMap) : {};
	} catch {
		return {};
	}
}

export function saveOverrides(overrides: ModelOverrideMap): void {
	try {
		localStorage.setItem(MODEL_OVERRIDES_LOCALSTORAGE_KEY, JSON.stringify(overrides));
	} catch {
		console.warn('[ModelsManager] Failed to persist model overrides');
	}
}

/** Backend a model is served by, the local server reads as "This server". */
export function servedByLabel(option: ModelOption): string {
	const backend = getBackend(option.backendId);

	if (!backend || backend.id === LOCAL_BACKEND_ID) return 'This server';

	return backend.name;
}

export function isLocalOption(option: ModelOption): boolean {
	return (option.backendId ?? LOCAL_BACKEND_ID) === LOCAL_BACKEND_ID;
}

/** File size of a local GGUF, when the router reported one. */
export function modelSizeLabel(option: ModelOption): string | null {
	const bytes = option.meta?.size;

	if (typeof bytes === 'number' && bytes > 0) return formatFileSize(bytes);

	return null;
}

export function modelParamsLabel(option: ModelOption): string | null {
	if (option.parsedId?.params) return option.parsedId.params;

	const params = option.meta?.n_params;

	return typeof params === 'number' ? formatParameters(params) : null;
}

export function modelQuantLabel(option: ModelOption): string | null {
	return option.parsedId?.quantization ?? null;
}

/** Extra args the router applies when this model is loaded. */
export function loadExtraArgs(override?: ModelOverride): string[] {
	const load = override?.load;

	if (!load) return [];

	const args: string[] = [];

	if (load.contextLength) args.push('--ctx-size', String(load.contextLength));

	if (load.gpuOffload !== undefined) args.push('--n-gpu-layers', String(load.gpuOffload));

	if (load.cpuThreads) args.push('--threads', String(load.cpuThreads));

	if (load.batchSize) args.push('--batch-size', String(load.batchSize));

	if (load.ubatchSize) args.push('--ubatch-size', String(load.ubatchSize));

	if (load.flashAttention) args.push('--flash-attn', 'on');

	if (load.useMmap === false) args.push('--no-mmap');

	if (load.keepInMemory === false) args.push('--no-kv-offload');

	return args;
}
