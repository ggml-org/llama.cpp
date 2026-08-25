import { HuggingFaceService } from '$lib/services';
import type { ModelManagerParent, ModelManagerQuant, ModelManagerQuantOrg } from '$lib/types';
import type { ModelOption } from '$lib/types/models';

/** Strip the `:quant` / `:quant-VARIANT` tag from a model id to get the repo id. */
export function getRepoId(modelId: string): string {
	return modelId.split(':')[0] ?? modelId;
}

/**
 * Normalize a model id to a parent grouping key: last path segment with the
 * container-format suffix (`-GGUF`, `-GGML`) stripped. `Qwen/Qwen3.8-27B` and
 * `ggml-org/Qwen3.8-27B-GGUF` both normalize to `Qwen3.8-27B`.
 */
export function normalizeParentName(id: string): string {
	const slashIdx = id.lastIndexOf('/');
	const name = slashIdx !== -1 ? id.slice(slashIdx + 1) : id;

	return name.replace(/-(GGUF|GGML)$/i, '');
}

/** Heuristic parent id for a repo: normalized repo name (org dropped). */
export function heuristicParentId(repoId: string): string {
	return normalizeParentName(repoId);
}

// Module-level cache so base models survive route navigation.
const baseModelCache = new Map<string, string | null>();

/**
 * Resolve the original (non-GGUF) model for a repo via HF metadata. Prefers
 * `cardData.base_model` (a string or a list), falling back to the
 * `base_model:<org>/<name>` tag. Cached per repo; null means "not found".
 */
export async function resolveBaseModel(repoId: string): Promise<string | null> {
	if (baseModelCache.has(repoId)) return baseModelCache.get(repoId) ?? null;

	let resolved: string | null = null;

	try {
		const details = await HuggingFaceService.getDetails(repoId);
		const base = details?.cardData?.base_model;
		const first = Array.isArray(base) ? base[0] : base;

		if (first && first.trim()) {
			resolved = first.trim();
		} else {
			// tag fallback, e.g. `base_model:Qwen/Qwen3-8B`
			const tag = details?.tags?.find(
				(t) => t.startsWith('base_model:') && !t.startsWith('base_model:quantized:')
			);
			const fromTag = tag?.slice('base_model:'.length).trim();

			if (fromTag) resolved = fromTag;
		}
	} catch {
		// fall through to the heuristic
	}

	baseModelCache.set(repoId, resolved);

	return resolved;
}

/**
 * Build the manager tree from installed models and resolved base models.
 * Repos are grouped under a parent keyed by the heuristic name; the display id
 * prefers the resolved HF base_model.
 */
export function buildModelManagerTree(
	models: ModelOption[],
	baseModels: ReadonlyMap<string, string | null>
): ModelManagerParent[] {
	// repoId -> quant -> { main, drafts, mmproj }
	const repoMap = new Map<string, Map<string, ModelManagerQuant>>();

	for (const option of models) {
		const parsed = option.parsedId;
		const repoId = getRepoId(option.model);
		const quant = parsed?.quantization ?? null;
		const variant = parsed?.variant ?? null;

		let quantMap = repoMap.get(repoId);

		if (!quantMap) {
			quantMap = new Map();
			repoMap.set(repoId, quantMap);
		}

		const key = quant ?? '';

		let entry = quantMap.get(key);

		if (!entry) {
			entry = { drafts: [], main: option, mmproj: null, quant };
			quantMap.set(key, entry);
		}

		if (variant === 'mmproj') {
			entry.mmproj = option;
		} else if (variant) {
			entry.drafts.push({ option, variant });
		} else {
			entry.main = option;
		}
	}

	// Build quant orgs.
	const quantOrgs: ModelManagerQuantOrg[] = [];

	for (const [repoId, quantMap] of repoMap) {
		const quants = Array.from(quantMap.values()).sort((a, b) =>
			(a.quant ?? '').localeCompare(b.quant ?? '')
		);

		quants.forEach((q) => q.drafts.sort((a, b) => a.variant.localeCompare(b.variant)));

		const slashIdx = repoId.indexOf('/');
		const orgName = slashIdx !== -1 ? repoId.slice(0, slashIdx) : repoId;

		quantOrgs.push({ orgName, quants, repoId });
	}

	// Group quant orgs under parents.
	const parentMap = new Map<string, ModelManagerParent>();

	for (const org of quantOrgs) {
		const resolved = baseModels.get(org.repoId) ?? null;
		// Key by the model name so repos of the same model group together even
		// when only some of them resolved a base_model.
		const key = normalizeParentName(resolved ?? org.repoId);

		let parent = parentMap.get(key);

		if (!parent) {
			parent = { parentId: resolved ?? key, quantOrgs: [] };
			parentMap.set(key, parent);
		} else if (resolved && parent.parentId === key) {
			// Prefer the resolved base_model for display once known.
			parent.parentId = resolved;
		}

		parent.quantOrgs.push(org);
	}

	const parents = Array.from(parentMap.values());

	parents.forEach((p) => p.quantOrgs.sort((a, b) => a.repoId.localeCompare(b.repoId)));
	parents.sort((a, b) => a.parentId.localeCompare(b.parentId));

	return parents;
}
