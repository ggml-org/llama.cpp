import { MODEL_ID } from '$lib/constants';
import { HuggingFaceService, ModelsService } from '$lib/services';
import { formatParameters, normalizeModelName } from '$lib/utils';

export interface UseModelParamsFallbackOptions {
	/** Model id to parse; null disables the fallback. */
	modelId: () => string | null | undefined;
	/** Server-reported parameter count from the model meta, when available. */
	metaParams?: () => unknown;
}

/**
 * Params badge fallback for model ids that carry no params token (`Kimi-K3`).
 * Prefers the meta `n_params`, then the HF GGUF total (`gguf.total`) fetched
 * lazily like the models hub list does, only when neither the id nor the meta
 * has it. Returns the formatted badge text, or undefined when unavailable.
 */
export function useModelParamsFallback(opts: UseModelParamsFallbackOptions) {
	const parsedParams = $derived.by(() => {
		const id = opts.modelId();

		return id ? ModelsService.parseModelId(id).params : null;
	});
	const metaParams = $derived.by(() => {
		const value = opts.metaParams?.();

		return typeof value === 'number' && value > 0 ? value : null;
	});
	// HF repo id: the `org/model` part without the `:revision/quant` suffix;
	// local paths and bare names are not HF repos.
	const hfRepoId = $derived.by(() => {
		const id = opts.modelId();

		if (!id) return null;

		const [repo] = id.split(MODEL_ID.QUANTIZATION_SEPARATOR);
		const normalized = normalizeModelName(repo ?? '');

		return normalized.includes(MODEL_ID.ORG_SEPARATOR) ? normalized : null;
	});

	let fetchedParams = $state<number | null>(null);

	$effect(() => {
		fetchedParams = null;

		if (parsedParams || metaParams || !hfRepoId) return;

		let cancelled = false;

		void HuggingFaceService.getDetails(hfRepoId).then((info) => {
			if (!cancelled && info?.gguf?.total) fetchedParams = info.gguf.total;
		});

		return () => {
			cancelled = true;
		};
	});

	const paramsFallback = $derived(
		!parsedParams && (metaParams ?? fetchedParams)
			? formatParameters(metaParams ?? fetchedParams)
			: undefined
	);

	return { paramsFallback };
}
