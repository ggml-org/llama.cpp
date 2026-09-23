import { LOCAL_BACKEND_ID, MODEL_OVERRIDES_LOCALSTORAGE_KEY } from '$lib/constants';
import { HuggingFaceService, ModelsService } from '$lib/services';
import type { ModelLoadProgress, ModelOption } from '$lib/types/models';
import { getBackend } from '$lib/utils/api-base';
import { formatFileSize, formatParameters } from '$lib/utils/formatters';
import { SvelteMap } from 'svelte/reactivity';

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

export type { ModelLoadProgress };

/** One repo of the table, with the quants it ships as its rows. */
export interface ModelQuantGroup {
	/** The quant that carries the identity of the repo. */
	base: ModelOption;
	key: string;
	quants: ModelOption[];
}

/** One collapsible block of the manager's table. */
export interface ModelsTableGroup {
	/** Null for the loaded and favorites groups. */
	backendId: string | null;
	isLocal: boolean;
	/** One entry per repo, its quants hanging off it. */
	items: ModelQuantGroup[];
	key: string;
	/** Picks the header icon; providers use their backend logo instead. */
	kind: 'favorites' | 'hidden' | 'loaded' | 'local' | 'provider';
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

/**
 * File size of the model's own quant. The router reports one for some backends;
 * otherwise a local GGUF reads it from its repo tree, the same source the
 * discovery details use. Returns null when neither knows.
 */
export async function resolveModelSize(option: ModelOption): Promise<string | null> {
	const reported = modelSizeLabel(option);

	if (reported) return reported;

	if (!isLocalOption(option)) return null;

	const [repo, quant] = option.model.split(':');

	if (!repo || !quant) return null;

	const tree = await HuggingFaceService.getTree(repo);
	const file = HuggingFaceService.collapseGgufShards(
		HuggingFaceService.filterByExtension(tree, '.gguf')
	).find((entry) => {
		const meta = HuggingFaceService.extractQuantMeta(entry.path);

		return meta?.quant === quant && !meta.sidecar;
	});

	return file?.size ? formatFileSize(file.size) : null;
}

/** Repo a `repo:quant` id belongs to, the id itself when it carries no quant. */
export function modelRepoKey(model: string): string {
	const quant = ModelsService.parseModelId(model).quantization;

	return quant ? model.slice(0, model.lastIndexOf(':')) : model;
}

/** Fold the quants of one repo into a single entry, so the table shows one row per model. */
export function groupModelQuants(models: ModelOption[]): ModelQuantGroup[] {
	const groups = new SvelteMap<string, ModelQuantGroup>();

	for (const option of models) {
		const key = modelRepoKey(option.model);
		const group = groups.get(key);

		if (group) {
			group.quants.push(option);

			continue;
		}

		groups.set(key, { base: option, key, quants: [option] });
	}

	return Array.from(groups.values());
}

/**
 * Family a repo belongs to, from its name. The version is dropped, so `Qwen3.5`,
 * `Qwen3.8-27B` and `Qwen3.8-Flash-Next` all read as `Qwen`. A name that does not
 * start with letters keeps its first segment.
 */
export function modelFamilyKey(repo: string): string {
	const name = repo.split('/').pop() ?? repo;
	const letters = name.match(/^[A-Za-z]+/);

	return letters ? letters[0] : (name.split(/[-_.]/)[0] ?? name);
}

/** One family of the table, the repos it covers. */
export interface ModelFamilyGroup {
	entries: ModelQuantGroup[];
	key: string;
	label: string;
}

/** Fold repos into families, so `Qwen3.8` collects its sizes and variants. */
export function groupModelFamilies(entries: ModelQuantGroup[]): ModelFamilyGroup[] {
	const families = new SvelteMap<string, ModelFamilyGroup>();

	for (const entry of entries) {
		const key = modelFamilyKey(entry.base.model);
		const family = families.get(key);

		if (family) {
			family.entries.push(entry);

			continue;
		}

		families.set(key, { entries: [entry], key, label: key });
	}

	return Array.from(families.values());
}

/** Compact "last used" label: minutes, hours, then days. */
export function formatLastUsed(timestamp?: number): string {
	if (!timestamp) return '—';

	const minutes = Math.floor((Date.now() - timestamp) / 60_000);

	if (minutes < 1) return 'just now';

	if (minutes < 60) return `${minutes}m`;

	const hours = Math.floor(minutes / 60);

	if (hours < 24) return `${hours}h`;

	return `${Math.floor(hours / 24)}d`;
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
