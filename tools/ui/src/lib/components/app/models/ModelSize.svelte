<script lang="ts">
	import { modelSizeLabel } from './ModelsManager/utils';
	import { HuggingFaceService } from '$lib/services';
	import type { ModelOption } from '$lib/types/models';
	import { formatFileSize } from '$lib/utils/formatters';

	interface Props {
		option: ModelOption;
	}

	let { option }: Props = $props();

	// the router reports a size for some backends, that wins over a lookup
	let reported = $derived(modelSizeLabel(option));
	let el = $state<HTMLElement | null>(null);
	let isNearViewport = $state(false);
	let fetchedBytes = $state<number | null>(null);

	// a repo tree costs one Hugging Face request, so wait until the row is near
	// the viewport and only look up local GGUF quants, which are files on disk
	let repo = $derived(option.model.split(':')[0] ?? '');
	let quant = $derived(option.model.split(':')[1] ?? '');

	$effect(() => {
		if (isNearViewport || !el) return;

		if (typeof IntersectionObserver === 'undefined') {
			isNearViewport = true;

			return;
		}

		const observer = new IntersectionObserver(
			(entries) => {
				if (entries.some((entry) => entry.isIntersecting)) {
					isNearViewport = true;
					observer.disconnect();
				}
			},
			{ rootMargin: '200px' }
		);

		observer.observe(el);

		return () => observer.disconnect();
	});

	$effect(() => {
		fetchedBytes = null;

		if (reported || !isNearViewport || !repo || !quant) return;

		let cancelled = false;

		void HuggingFaceService.getTree(repo)
			.then((tree) => {
				if (cancelled) return;

				// shards of one quant collapse into a single entry carrying the total
				const file = HuggingFaceService.collapseGgufShards(
					HuggingFaceService.filterByExtension(tree, '.gguf')
				).find((entry) => {
					const meta = HuggingFaceService.extractQuantMeta(entry.path);

					return meta?.quant === quant && !meta.sidecar;
				});

				if (file?.size) fetchedBytes = file.size;
			})
			// best-effort: offline or a repo we cannot read keeps the dash
			.catch(() => {});

		return () => {
			cancelled = true;
		};
	});

	let label = $derived(reported ?? (fetchedBytes ? formatFileSize(fetchedBytes) : '—'));
</script>

<span bind:this={el} class="text-sm text-muted-foreground">{label}</span>
