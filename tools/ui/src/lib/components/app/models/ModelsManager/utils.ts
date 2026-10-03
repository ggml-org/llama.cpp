import { LOCAL_BACKEND_ID } from '$lib/constants';
import { ModelCapability, ModelGroupKind, ModelsTableGroupKind } from '$lib/enums';
import { HuggingFaceService, ModelsService } from '$lib/services';
import { modelsStore } from '$lib/stores';
import type { ModelDownloadEntry, ModelDownloadProgress, ModelOption } from '$lib/types/models';
import { detectThinkingSupport, detectToolUseSupport, repoOf } from '$lib/utils';

/** One repo of the table, with the rows it ships as. */
export interface ModelQuantGroup {
	base: ModelOption;
	key: string;
	kind: ModelGroupKind;
	quants: ModelOption[];
}

/** Byte counts of a tracked download: live while it runs, frozen while paused. */
export function downloadProgressFor(repoWithTag: string): ModelDownloadProgress | null {
	return (
		modelsStore.status.getDownloadProgress(repoWithTag) ??
		modelsStore.status.getPausedDownloadProgress(repoWithTag)
	);
}

/**
 * One table row per tracked download, so the table lists them the way it lists
 * any other model. A paused download is already in the router's listing, so its
 * own option is reused; a fresh one stands for its tag alone.
 */
export function downloadGroups(
	entries: ModelDownloadEntry[],
	models: ModelOption[]
): ModelQuantGroup[] {
	return entries.map((entry) => {
		const option = models.find((candidate) => candidate.model === entry.repoWithTag) ?? {
			backendId: LOCAL_BACKEND_ID,
			capabilities: [],
			id: entry.repoWithTag,
			model: entry.repoWithTag,
			name: entry.repoWithTag
		};

		return {
			base: option,
			key: `download::${entry.repoWithTag}`,
			kind: ModelGroupKind.QUANTS,
			quants: [option]
		};
	});
}

/** One collapsible block of the manager's table. */
export interface ModelsTableGroup {
	/** One entry per repo, its quants hanging off it. */
	items: ModelQuantGroup[];
	key: string;
	kind: ModelsTableGroupKind;
	label: string;
}

/** Context the model runs with: what a loaded model reports. */
export function configuredContext(option: ModelOption): number | null {
	return modelsStore.isModelRunning(option.model)
		? modelsStore.props.getModelContextSize(option.model)
		: null;
}

/**
 * Capability a model reports: true or false once its listing or its chat template
 * answers, null while the Hub record that carries the template is not read yet.
 */
export function modelCapability(option: ModelOption, capability: ModelCapability): boolean | null {
	if (option.capabilities.includes(capability)) return true;

	const details = HuggingFaceService.cachedDetails(option.model);

	// only the chat template answers this, and it arrives with the Hub record
	if (details === undefined) return null;

	const template = details?.gguf?.chat_template ?? '';

	return capability === ModelCapability.TOOL_USE
		? detectToolUseSupport(template)
		: detectThinkingSupport(template);
}

/**
 * Context a model reports: the provider listing first, then the cached Hub record.
 * Null while neither of them has answered.
 */
export function modelContextLength(option: ModelOption): number | null {
	return (
		option.contextLength ??
		HuggingFaceService.cachedDetails(option.model)?.gguf?.context_length ??
		null
	);
}

/** Fold the quants of one repo into a single entry, so the table shows one row per model. */
export function groupModelQuants(models: ModelOption[]): ModelQuantGroup[] {
	const groups = new Map<string, ModelQuantGroup>();

	for (const option of models) {
		const key = repoOf(option.model);
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

/**
 * Split the repos of a section into the quants that are hidden and the ones that
 * are not, so a partly hidden repo lists in both blocks instead of dragging its
 * visible quants into the hidden one.
 */
export function splitHiddenQuants(
	entries: ModelQuantGroup[],
	isHidden: (option: ModelOption) => boolean
): { hidden: ModelQuantGroup[]; local: ModelQuantGroup[] } {
	const hidden: ModelQuantGroup[] = [];
	const local: ModelQuantGroup[] = [];

	for (const entry of entries) {
		const hiddenQuants = entry.quants.filter((quant) => isHidden(quant));
		const localQuants = entry.quants.filter((quant) => !isHidden(quant));

		if (hiddenQuants.length > 0) {
			hidden.push({
				...entry,
				base: hiddenQuants[0],
				key: `${entry.key}::hidden`,
				quants: hiddenQuants
			});
		}

		if (localQuants.length > 0) {
			local.push({ ...entry, base: localQuants[0], quants: localQuants });
		}
	}

	return { hidden, local };
}
