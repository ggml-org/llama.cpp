<script lang="ts">
	import { MessageSquareText } from '@lucide/svelte';
	import { DialogSystemMessageEdit } from '#lib/components/app/index.ts';
	import { SYSTEM_MESSAGE_PLACEHOLDER } from '#lib/constants/index.ts';
	import { chatStore } from '#lib/stores/index.ts';

	interface Props {
		content: string;
		disabled?: boolean;
	}

	let { content, disabled = false }: Props = $props();

	let isOpen = $state(false);

	// a placeholder is a system message nobody has written yet
	const hasContent = $derived(content.trim() !== '' && content !== SYSTEM_MESSAGE_PLACEHOLDER);
</script>

{#if hasContent}
	<div class="grid">
		<button
			class="mt-1.5 flex w-auto min-w-0 items-center gap-1 rounded-md px-2 py-1 text-caption text-muted-foreground backdrop-blur-2xl hover:text-foreground"
			{disabled}
			onclick={() => (isOpen = true)}
			title={content}
			type="button"
		>
			<MessageSquareText class="size-icon-sm" />

			<span class="truncate">Custom system message</span>
		</button>
	</div>

	<DialogSystemMessageEdit
		bind:open={isOpen}
		{content}
		onSave={(next) => chatStore.saveSystemPrompt(next)}
	/>
{/if}
