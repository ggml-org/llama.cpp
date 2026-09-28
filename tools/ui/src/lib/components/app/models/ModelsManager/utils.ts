import { ModelCapability, ServerModelStatus } from '$lib/enums';
import { HuggingFaceService, ModelsService } from '$lib/services';
import { modelsStore } from '$lib/stores';
import type { ModelModalities, ModelOption } from '$lib/types/models';
import { detectThinkingSupport, detectToolUseSupport } from '$lib/utils';

/** What a group folds, which decides its label. */
export type ModelGroupKind = 'quants' | 'variants';

/** One repo of the table, with the rows it ships as. */
export interface ModelQuantGroup {
	base: ModelOption;
	key: string;
	kind: ModelGroupKind;
	quants: ModelOption[];
}

/**
 * Kind of one collapsible block of the manager's table. A plain constant map, so
 * provider blocks can extend the union with their own string kinds.
 */
export const ModelsTableGroupKind = {
	FAVORITES: 'favorites',
	HIDDEN: 'hidden',
	LOADED: 'loaded',
	LOCAL: 'local'
} as const;

export type ModelsTableGroupKind = (typeof ModelsTableGroupKind)[keyof typeof ModelsTableGroupKind];

/** Column the manager's table can be ordered by. */
export const ModelsTableSortKey = {
	CONTEXT: 'context',
	NAME: 'name',
	STATUS: 'status'
} as const;

export type ModelsTableSortKey = (typeof ModelsTableSortKey)[keyof typeof ModelsTableSortKey];

/** Header label of each manager section. */
export const MODELS_TABLE_GROUP_LABELS: Record<ModelsTableGroupKind, string> = {
	[ModelsTableGroupKind.FAVORITES]: 'Favorites',
	[ModelsTableGroupKind.HIDDEN]: 'Hidden models',
	[ModelsTableGroupKind.LOADED]: 'Loaded models',
	[ModelsTableGroupKind.LOCAL]: 'Local models'
};

/** One collapsible block of the manager's table. */
export interface ModelsTableGroup {
	/** One entry per repo, its quants hanging off it. */
	items: ModelQuantGroup[];
	key: string;
	kind: ModelsTableGroupKind;
	label: string;
}

/** Modalities a model can accept, as the manager filter offers them. */
export type ModalityKey = keyof ModelModalities;

/** Modalities the manager filter offers, in display order. */
export const MODALITY_KEYS: ModalityKey[] = ['vision', 'video', 'audio'];

/** True when the model is loaded (or sleeping) and not mid-operation. */
export function isModelRunning(option: ModelOption): boolean {
	const status = modelsStore.getModelStatus(option.model);

	return (
		(status === ServerModelStatus.LOADED || status === ServerModelStatus.SLEEPING) &&
		!modelsStore.status.isOperationInProgress(option.model)
	);
}

/** Context the model runs with: what a loaded model reports. */
export function configuredContext(option: ModelOption): number | null {
	return isModelRunning(option) ? modelsStore.props.getModelContextSize(option.model) : null;
}

export function modelSupports(option: ModelOption, capability: ModelCapability): boolean {
	if (option.capabilities.includes(capability)) return true;

	const template = HuggingFaceService.cachedDetails(option.model)?.gguf?.chat_template ?? '';

	return capability === ModelCapability.TOOL_USE
		? detectToolUseSupport(template)
		: detectThinkingSupport(template);
}

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

/** Repo a `repo:quant` id belongs to, the id itself when it carries no quant. */
export function modelRepoKey(model: string): string {
	const quant = ModelsService.parseModelId(model).quantization;

	return quant ? model.slice(0, model.lastIndexOf(':')) : model;
}

/** Fold the quants of one repo into a single entry, so the table shows one row per model. */
export function groupModelQuants(models: ModelOption[]): ModelQuantGroup[] {
	const groups = new Map<string, ModelQuantGroup>();

	for (const option of models) {
		const key = modelRepoKey(option.model);
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

		// the very same id twice is not a quant set; keep those rows apart
		if (modelIds.length > 1 && modelIds.every((model) => model === modelIds[0])) {
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

/** What a group folds: quants or variants of one repo. */
function groupKind(quants: ModelOption[]): ModelGroupKind {
	const isQuant = quants.every(
		(option) => (option.parsedId ?? ModelsService.parseModelId(option.model)).quantization
	);

	return isQuant ? 'quants' : 'variants';
}
