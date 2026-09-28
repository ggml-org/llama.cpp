<script lang="ts">
	import { isLocalOption } from './ModelsManager/utils';
	import { HuggingFaceService } from '$lib/services';
	import type { ModelOption } from '$lib/types/models';
	import { formatParameters } from '$lib/utils/formatters';

	interface Props {
		class?: string;
		/** Context the model is set to run with; paired with the supported one. */
		configured?: number | null;
		option: ModelOption;
	}

	let { class: className = '', configured = null, option }: Props = $props();

	// a listing that reports a context window (OpenRouter, Groq, HF) needs no lookup
	let reported = $derived(option.contextLength ?? null);
	let el = $state<HTMLElement | null>(null);
	let isNearViewport = $state(false);
	let fetched = $state<number | null>(null);

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
		fetched = null;

		if (reported || !isNearViewport || !isLocalOption(option)) return;

		// a local GGUF carries its trained context in the model metadata
		const repo = option.model.split(':')[0] ?? '';

		if (!repo) return;

		let cancelled = false;

		void HuggingFaceService.getDetails(repo)
			.then((details) => {
				if (!cancelled) fetched = details?.gguf?.context_length ?? null;
			})
			// best-effort: offline or a repo we cannot read keeps the dash
			.catch(() => {});

		return () => {
			cancelled = true;
		};
	});

	let context = $derived(reported ?? fetched);
	// `configured / supported`, or whichever of the two is known
	let values = $derived(
		[configured, context].filter((value): value is number => typeof value === 'number')
	);
</script>

<span bind:this={el} class={['text-sm text-muted-foreground', className]}>
	{values.length ? `${values.map((value) => formatParameters(value)).join(' / ')} tokens` : '—'}
</span>
