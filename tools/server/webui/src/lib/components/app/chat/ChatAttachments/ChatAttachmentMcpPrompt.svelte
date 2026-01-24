<script lang="ts">
	import { ChatMessageMcpPromptContent, RemoveButton } from '$lib/components/app';
	import type { DatabaseMessageExtraMcpPrompt } from '$lib/types';

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
	<ChatMessageMcpPromptContent {prompt} variant="attachment" {isLoading} {loadError} />

	{#if !readonly && onRemove}
		<div
			class="absolute top-8 right-2 flex items-center justify-center opacity-0 transition-opacity group-hover:opacity-100"
		>
			<RemoveButton id={prompt.name} onRemove={() => onRemove?.()} />
		</div>
	{/if}
</div>
