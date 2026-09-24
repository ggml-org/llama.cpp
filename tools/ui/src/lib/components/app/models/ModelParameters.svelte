<script lang="ts">
	import { isLocalOption, modelParamsLabel } from './ModelsManager/utils';
	import { HuggingFaceService } from '$lib/services';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		class?: string;
		option: ModelOption;
	}

	let { class: className = '', option }: Props = $props();

	let el = $state<HTMLElement | null>(null);
	let isNearViewport = $state(false);
	let total = $state<number | null>(null);

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
		total = null;

		// nothing to look up when the id or the listing already carries the count
		if (!isNearViewport || !isLocalOption(option)) return;

		if (option.parsedId?.params || typeof option.meta?.n_params === 'number') return;

		// a local GGUF names no parameter count, but the Hub knows the model's
		const repo = option.model.split(':')[0] ?? '';

		if (!repo.includes('/')) return;

		let cancelled = false;

		void HuggingFaceService.getDetails(repo)
			.then((details) => {
				if (!cancelled) total = details?.gguf?.total ?? null;
			})
			// best-effort: offline or a repo we cannot read keeps the dash
			.catch(() => {});

		return () => {
			cancelled = true;
		};
	});

	let label = $derived(modelParamsLabel(option, total));
</script>

<span bind:this={el} class={['text-sm text-muted-foreground', className]}>{label ?? '—'}</span>
