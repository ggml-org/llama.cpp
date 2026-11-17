<!--
	Llama UI's ModelOrgAvatar, shown only while the Hub metadata setting is on.
-->
<script lang="ts">
	import { ModelOrgAvatar as LlamaUiModelOrgAvatar } from '@ggml-org/llama-ui/models';
	import { HuggingFaceService } from '#lib/services/index.ts';
	import type { ComponentProps } from 'svelte';

	let props: ComponentProps<typeof LlamaUiModelOrgAvatar> = $props();

	// With the Hub metadata setting off there is nothing to fetch and nothing to
	// show: the avatar is hidden entirely instead of falling back to a monogram.
	let hubEnabled = $derived(HuggingFaceService.isEnabled());
</script>

{#if hubEnabled}
	<LlamaUiModelOrgAvatar {...props} />
{/if}
