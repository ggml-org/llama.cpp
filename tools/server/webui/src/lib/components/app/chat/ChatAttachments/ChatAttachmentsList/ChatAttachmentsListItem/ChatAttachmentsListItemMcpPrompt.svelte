<script lang="ts">
	import { ChatMessageMcpPromptContent, ActionIcon } from '$lib/components/app';
	import { X } from '@lucide/svelte';
	import type { DatabaseMessageExtraMcpPrompt } from '$lib/types';
	import { McpPromptVariant } from '$lib/enums';

	interface Props {
		class?: string;
		prompt: DatabaseMessageExtraMcpPrompt;
		readonly?: boolean;
		isLoading?: boolean;
		loadError?: string;
		onRemove?: () => void;
	}

	let {
		class: className = '',
		prompt,
		readonly = false,
		isLoading = false,
		loadError,
		onRemove
	}: Props = $props();
</script>

<div class="group relative {className}">
	<ChatMessageMcpPromptContent
		{prompt}
		variant={McpPromptVariant.ATTACHMENT}
		{isLoading}
		{loadError}
	/>

	{#if !readonly && onRemove}
		<div
			class="absolute top-10 right-2 flex items-center justify-center opacity-0 transition-opacity group-hover:opacity-100"
		>
			<ActionIcon icon={X} tooltip="Remove" stopPropagationOnClick onclick={() => onRemove?.()} />
		</div>
	{/if}
</div>
