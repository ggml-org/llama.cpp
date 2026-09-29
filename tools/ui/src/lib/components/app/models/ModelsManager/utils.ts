import { ModelGroupKind, ModelsTableGroupKind } from '$lib/constants';
import { ModelCapability, ServerModelStatus } from '$lib/enums';
import { HuggingFaceService, ModelsService } from '$lib/services';
import { modelsStore } from '$lib/stores';
import type { ModelOption } from '$lib/types/models';
import { detectThinkingSupport, detectToolUseSupport } from '$lib/utils';

/** One repo of the table, with the rows it ships as. */
export interface ModelQuantGroup {
	base: ModelOption;
	key: string;
	kind: ModelGroupKind;
	quants: ModelOption[];
}

/** One collapsible block of the manager's table. */
export interface ModelsTableGroup {
	/** One entry per repo, its quants hanging off it. */
	items: ModelQuantGroup[];
	key: string;
	kind: ModelsTableGroupKind;
	label: string;
}

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

/** Context a model reports: the provider listing first, then the cached Hub record. */
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

		groups.set(key, { base: option, key, kind: ModelGroupKind.QUANTS, quants: [option] });
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
				kind: ModelGroupKind.VARIANTS,
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

	return isQuant ? ModelGroupKind.QUANTS : ModelGroupKind.VARIANTS;
}
