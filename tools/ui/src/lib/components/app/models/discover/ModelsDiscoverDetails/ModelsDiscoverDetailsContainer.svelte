<script lang="ts">
	import ModelsDiscoverDetails from './ModelsDiscoverDetails.svelte';
	import { HuggingFaceService } from '$lib/services';
	import type { HfModelDetailInfo, HfModelSibling } from '$lib/types/huggingface';

	interface Props {
		/** Full HuggingFace model id, e.g. `ggml-org/gemma-3-4b-it-GGUF`. */
		modelId: string;
	}

	let { modelId }: Props = $props();

	let details = $state<HfModelDetailInfo | null>(null);
	let files = $state<HfModelSibling[]>([]);
	let readme = $state<string | null>(null);
	let loading = $state(false);
	let error = $state<string | null>(null);

	$effect(() => {
		const id = modelId;

		if (!id) return;

		let cancelled = false;

		loading = true;
		error = null;

		void (async () => {
			try {
				const [info, tree, readmeText] = await Promise.all([
					HuggingFaceService.getDetails(id),
					HuggingFaceService.getTree(id),
					HuggingFaceService.getReadme(id)
				]);

				if (cancelled) return;

				if (!info) {
					error = 'Model not found';

					return;
				}

				details = info;
				files = HuggingFaceService.filterByExtension(
					HuggingFaceService.collapseGgufShards(tree),
					'.gguf'
				);
				readme = readmeText;
			} catch (err) {
				if (cancelled) return;

				error = err instanceof Error ? err.message : 'Failed to load model';
			} finally {
				if (!cancelled) loading = false;
			}
		})();

		return () => {
			cancelled = true;
		};
	});
</script>

<ModelsDiscoverDetails {details} {error} {files} {loading} {modelId} {readme} />
