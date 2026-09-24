import {
	LOCAL_BACKEND_ID,
	MODEL_ID,
	MODEL_OVERRIDES_LOCALSTORAGE_KEY,
	type ModelSidecar,
	SETTINGS_KEYS,
	SPEC_TYPE
} from '$lib/constants';
import { HuggingFaceService, ModelsService } from '$lib/services';
import { settingsStore } from '$lib/stores';
import type { ModelLoadProgress, ModelModalities, ModelOption } from '$lib/types/models';
import { getBackend } from '$lib/utils/api-base';
import { formatFileSize, formatParameters } from '$lib/utils/formatters';
import { rawModelId } from '$lib/utils/model-option-id';
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

/** What a group folds, which decides its label. */
export type ModelGroupKind = 'providers' | 'quants' | 'variants';

/** One repo of the table, with the rows it ships as. */
export interface ModelQuantGroup {
	base: ModelOption;
	key: string;
	kind: ModelGroupKind;
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
	kind: 'compat' | 'favorites' | 'hidden' | 'loaded' | 'local' | 'provider';
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
/** A draft a model can speculate with, and whether the load settings point at it. */
export interface ModelDraft {
	/** The draft the settings name, which is the one a load would use. */
	active: boolean;
	kind: ModelSidecar | null;
	/** Qualified id of a draft from another repo; null when the file sits in this repo. */
	model: string | null;
}

/** Repo an id belongs to: the id without its quant tag. */
function repoOf(modelId: string): string | null {
	const raw = rawModelId(modelId).split(MODEL_ID.QUANTIZATION_SEPARATOR)[0] ?? '';

	return raw || null;
}

/** Sidecar a `--spec-type` value names, e.g. `draft-mtp` -> mtp. */
function sidecarFromSpecType(specType: string | null | undefined): ModelSidecar | null {
	if (!specType) return null;

	const entry = Object.entries(SPEC_TYPE).find(([, value]) => value === specType);

	return (entry?.[0] as ModelSidecar | undefined) ?? null;
}

/**
 * Repo a draft path names. A Hub cache path carries it (`models--org--name`), a plain
 * file next to the model does not, so that falls back to the file name.
 */
function repoFromDraftPath(path: string): string {
	const cached = /models--([^/]+)[/]/.exec(path);

	if (cached) {
		const [org, ...rest] = cached[1].split('--');

		if (rest.length > 0) return `${org}/${rest.join('--')}`;
	}

	return path.split(/[/]/).pop() ?? path;
}

/**
 * Draft the server's own launch arguments point at. The router reports the arguments a
 * model loads with, so this is what a load would really speculate with.
 */
export function draftFromArgs(args: string[] | undefined): ModelDraft | null {
	const flag = args?.indexOf('--model-draft') ?? -1;
	const path = flag === -1 ? null : (args?.[flag + 1] ?? null);

	if (!path) return null;

	return {
		active: true,
		kind: sidecarFromSpecType(args?.[(args?.indexOf('--spec-type') ?? -1) + 1]),
		model: repoFromDraftPath(path)
	};
}

/**
 * Drafts to show for a model: the one the load settings name, plus any draft sidecar
 * the model's own repo ships. The configured one is the active draft; a sidecar that
 * is merely on disk stays visible but idle.
 */
export function modelDrafts(
	option: ModelOption,
	available: ModelSidecar[] = [],
	configured?: string | null,
	configuredKind?: ModelSidecar | null
): ModelDraft[] {
	const value = configured?.trim() ?? '';
	const drafts: ModelDraft[] = [];

	if (value && value !== 'off') {
		const parsed = ModelsService.parseModelId(value);
		const sameRepo = repoOf(value) === repoOf(option.model);

		drafts.push({
			active: true,
			kind: parsed.sidecar ?? configuredKind ?? null,
			model: sameRepo ? null : value
		});
	}

	for (const kind of available) {
		const alreadyActive = drafts.some((draft) => draft.active && draft.kind === kind);

		if (alreadyActive) continue;

		drafts.push({ active: false, kind, model: null });
	}

	return drafts;
}

export function servedByLabel(option: ModelOption): string {
	const backend = getBackend(option.backendId);

	if (!backend || backend.id === LOCAL_BACKEND_ID) return 'This server';

	return backend.name;
}

/** Modalities a model can accept, as the manager filter offers them. */
export type ModalityKey = keyof ModelModalities;

/**
 * Context a model reports: the provider listing first, then whatever the Hub has
 * cached for it. 0 means nothing is known yet.
 */
export function modelContextLength(option: ModelOption): number {
	return (
		option.contextLength ??
		HuggingFaceService.cachedDetails(option.model)?.gguf?.context_length ??
		0
	);
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

	// the repo tree lookup only happens for installs that opted into the Hub
	if (!isLocalOption(option) || !settingsStore.config[SETTINGS_KEYS.ENABLE_DISCOVER_MODELS]) {
		return null;
	}

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
export function groupModelQuants(models: ModelOption[], mergeProviders = false): ModelQuantGroup[] {
	const groups = new SvelteMap<string, ModelQuantGroup>();

	for (const option of models) {
		const repo = modelRepoKey(option.model);
		// groups stay within one backend, so the same repo served by two providers
		// is not read as two quants of one model. The OAI-compat block asks for the
		// opposite: one repo, one row per provider that serves it.
		const key = mergeProviders ? repo : `${option.backendId ?? ''}::${repo}`;
		const group = groups.get(key);

		if (group) {
			group.quants.push(option);

			continue;
		}

		groups.set(key, { base: option, key, kind: 'quants', quants: [option] });
	}

	return Array.from(groups.values()).flatMap((group) => {
		const kind = groupKind(group.quants);
		const modelIds = group.quants.map((option) => option.model);

		// the very same id twice is not a quant set; keep those rows apart, unless
		// the group exists to list the providers that serve it
		if (
			kind !== 'providers' &&
			modelIds.length > 1 &&
			modelIds.every((model) => model === modelIds[0])
		) {
			return group.quants.map((option) => ({
				...group,
				base: option,
				key: `${group.key}::${option.id}`,
				kind: 'variants' as const,
				quants: [option]
			}));
		}

		return [{ ...group, kind }];
	});
}

/** What a group folds: providers, quants, or variants of one repo. */
function groupKind(quants: ModelOption[]): ModelGroupKind {
	const backends = new Set(quants.map((option) => option.backendId ?? ''));

	if (backends.size > 1) return 'providers';

	const isQuant = quants.every(
		(option) => (option.parsedId ?? ModelsService.parseModelId(option.model)).quantization
	);

	return isQuant ? 'quants' : 'variants';
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
