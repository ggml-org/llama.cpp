<script lang="ts">
	import ModelCapabilityIcons from './ModelCapabilityIcons.svelte';
	import { MODEL_ID } from '$lib/constants';
	import { ModelCapability } from '$lib/enums';
	import { HuggingFaceService } from '$lib/services';
	import type { ModelOption } from '$lib/types/models';
	import { detectThinkingSupport, detectToolUseSupport, nearViewport, repoOf } from '$lib/utils';

	interface Props {
		class?: string;
		/** Hide the modality icons, keeping the capability ones. */
		hideModalities?: boolean;
		option: ModelOption;
	}

	let { class: className = '', hideModalities = false, option }: Props = $props();

	// a listing that declares its capabilities (Ollama-compatible backends) needs no lookup
	let declared = $derived(option.capabilities);
	let isNearViewport = $state(false);
	let template = $state('');

	$effect(() => {
		template = '';

		if (!isNearViewport) return;

		// only the chat template says whether tools and reasoning work, and the Hub
		// carries it for a local GGUF whether the model is loaded or not
		const repo = repoOf(option.model);

		if (!repo.includes(MODEL_ID.ORG_SEPARATOR)) return;

		let cancelled = false;

		void HuggingFaceService.getDetails(repo)
			.then((details) => {
				if (!cancelled) template = details?.gguf?.chat_template ?? '';
			})
			// best-effort: offline or a repo we cannot read leaves the icons off
			.catch(() => {});

		return () => {
			cancelled = true;
		};
	});

	let supportsThinking = $derived(
		declared.includes(ModelCapability.REASONING) || detectThinkingSupport(template)
	);
	let supportsToolUse = $derived(
		declared.includes(ModelCapability.TOOL_USE) || detectToolUseSupport(template)
	);
</script>

<span use:nearViewport={() => (isNearViewport = true)} class={className}>
	<ModelCapabilityIcons
		{hideModalities}
		modalities={option.modalities}
		{supportsThinking}
		{supportsToolUse}
	/>
</span>
