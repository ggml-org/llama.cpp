<script lang="ts">
	import { Card } from '@ggml-org/llama-ui/ui/card';
	import { MessageSquareText, X } from '@lucide/svelte';
	import { ActionIcon, DialogSystemMessageEdit } from '#lib/components/app/index.ts';
	import { chatStore } from '#lib/stores/index.ts';

	interface Props {
		class?: string;
		content: string;
		disabled?: boolean;
	}

	let { class: className = '', content, disabled = false }: Props = $props();

	let isOpen = $state(false);
</script>

<div
	class="group relative flex max-w-75 min-w-50 flex-shrink-0 flex-col gap-2 first:ml-4 last:mr-4 {className}"
>
	<div class="inline-flex items-center gap-1.25 text-caption text-muted-foreground">
		<MessageSquareText class="size-3.5 shrink-0" />

		<span>Custom system message</span>
	</div>

	<button
		aria-label="Edit custom system message"
		class="text-left"
		{disabled}
		onclick={() => (isOpen = true)}
		type="button"
	>
		<!-- the system message bubble's dashed border and muted fill -->
		<Card
			class="overflow-hidden rounded-2xl !border-2 !border-dashed !border-border/50 bg-muted py-0 text-foreground"
		>
			<div
				class="overflow-y-auto px-3 py-2"
				style="max-height: 6rem; overflow-wrap: anywhere; word-break: break-word;"
			>
				<span class="text-caption whitespace-pre-wrap">{content}</span>
			</div>
		</Card>
	</button>

	{#if !disabled}
		<div
			class="absolute top-10 right-2 flex items-center justify-center opacity-0 transition-opacity group-hover:opacity-100"
		>
			<ActionIcon
				icon={X}
				onclick={() => chatStore.removeSystemPrompt()}
				stopPropagationOnClick
				tooltip="Remove"
			/>
		</div>
	{/if}
</div>

<DialogSystemMessageEdit
	bind:open={isOpen}
	{content}
	onSave={(next) => chatStore.saveSystemPrompt(next)}
/>
