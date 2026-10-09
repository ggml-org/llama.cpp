<script lang="ts">
	import { MODEL_ID } from '$lib/constants';
	import { HuggingFaceService } from '$lib/services';
	import type { ModelOption } from '$lib/types/models';
	import { nearViewport, repoOf } from '$lib/utils';
	import { formatContextLength } from '$lib/utils/formatters';

	interface Props {
		class?: string;
		/** Context the model is set to run with; paired with the supported one. */
		configured?: number | null;
		option: ModelOption;
	}

	let { class: className = '', configured = null, option }: Props = $props();

	// a listing that reports a context window (OpenRouter, Groq, HF) needs no lookup
	let reported = $derived(option.contextLength ?? null);
	let isNearViewport = $state(false);
	let fetched = $state<number | null>(null);

	$effect(() => {
		fetched = null;

		if (reported || !isNearViewport) return;

		// a local GGUF carries its trained context in the model metadata
		const repo = repoOf(option.model);

		// a local file path has no repo to read, and the Hub would answer 404
		if (!repo.includes(MODEL_ID.ORG_SEPARATOR)) return;

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
	let values = $derived(
		[configured, context].filter((value): value is number => typeof value === 'number')
	);
</script>

<span
	use:nearViewport={() => (isNearViewport = true)}
	class={['text-sm text-muted-foreground', className]}
>
	{formatContextLength(values)}
</span>
