<script lang="ts">
	import { Card } from '$lib/components/ui/card';
	import { ActionIcon } from '$lib/components/app';
	import { fadeInView } from '$lib/actions/fade-in-view.svelte';
	import { Send, Edit, Trash2 } from '@lucide/svelte';

	interface Props {
		class?: string;
		content: string;
		onSendImmediately: () => void;
		onEdit: (newContent: string) => void;
		onDelete: () => void;
		showProcessingInfo?: boolean;
	}

	let {
		class: className = '',
		content,
		onSendImmediately,
		onEdit,
		onDelete,
		showProcessingInfo = false
	}: Props = $props();

	let isMultiline = $state(false);
	let isEditing = $state(false);
	let editedContent = $state('');
	let messageElement: HTMLElement | undefined = $state();
	let textareaElement: HTMLTextAreaElement | undefined = $state();

	$effect(() => {
		if (!messageElement || !content.trim()) return;

		if (content.includes('\n')) {
			isMultiline = true;
			return;
		}

		const resizeObserver = new ResizeObserver((entries) => {
			for (const entry of entries) {
				const element = entry.target as HTMLElement;
				const estimatedSingleLineHeight = 24;
				isMultiline = element.offsetHeight > estimatedSingleLineHeight * 1.5;
			}
		});

		resizeObserver.observe(messageElement);

		return () => {
			resizeObserver.disconnect();
		};
	});

	function autoResizeTextarea() {
		if (!textareaElement) return;
		textareaElement.style.height = 'auto';
		textareaElement.style.height = `${textareaElement.scrollHeight}px`;
	}

	function handleEdit() {
		editedContent = content;
		isEditing = true;
		setTimeout(() => {
			if (textareaElement) {
				textareaElement.focus();
				textareaElement.setSelectionRange(
					textareaElement.value.length,
					textareaElement.value.length
				);
				autoResizeTextarea();
			}
		}, 0);
	}

	function handleSaveEdit() {
		const trimmed = editedContent.trim();
		if (trimmed) {
			onEdit(trimmed);
		}
		isEditing = false;
	}

	function handleCancelEdit() {
		isEditing = false;
	}

	function handleEditKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			handleSaveEdit();
		} else if (event.key === 'Escape') {
			event.preventDefault();
			handleCancelEdit();
		}
	}
</script>

<div
	use:fadeInView
	aria-label="Pending user message"
	class="group flex flex-col items-end gap-3 transition-opacity hover:opacity-80 md:gap-2 {className} sticky {showProcessingInfo
		? 'bottom-40'
		: 'bottom-32'}"
	role="group"
>
	{#if isEditing}
		<Card
			class="w-full max-w-36 rounded-[1.125rem] border-none bg-primary/5 px-3.75 py-2.5 text-muted-foreground backdrop-blur-md dark:bg-primary/8"
			style="overflow-wrap: anywhere; word-break: break-word;"
		>
			<textarea
				bind:this={textareaElement}
				bind:value={editedContent}
				onkeydown={handleEditKeydown}
				oninput={autoResizeTextarea}
				class="text-md w-full resize-none bg-transparent outline-none"
				style="overflow: hidden;"
			></textarea>
			<div class="mt-2 flex justify-end gap-2">
				<button
					class="rounded-md px-3 py-1 text-xs text-muted-foreground hover:bg-muted"
					onclick={handleCancelEdit}
				>
					Cancel
				</button>
				<button
					class="rounded-md bg-primary/10 px-3 py-1 text-xs text-foreground hover:bg-primary/20"
					onclick={handleSaveEdit}
				>
					Save
				</button>
			</div>
		</Card>
	{:else}
		{#if content.trim()}
			<Card
				class="max-w-[80%] overflow-y-auto rounded-[1.125rem] border-none bg-primary/5 px-3.75 py-1.5 text-muted-foreground backdrop-blur-md data-[multiline]:py-2.5 dark:bg-primary/8"
				data-multiline={isMultiline ? '' : undefined}
				style="overflow-wrap: anywhere; word-break: break-word;"
			>
				<span bind:this={messageElement} class="text-md whitespace-pre-wrap">
					{content}
				</span>
			</Card>
		{/if}

		<div class="max-w-[80%]">
			<div class="relative flex h-6 items-center justify-between">
				<div class="right-0 flex items-center gap-2 opacity-100 transition-opacity">
					<div
						class="pointer-events-auto inset-0 flex items-center gap-1 opacity-0 transition-all duration-150 group-hover:opacity-100"
					>
						<ActionIcon icon={Edit} tooltip="Edit" onclick={handleEdit} />
						<ActionIcon icon={Trash2} tooltip="Delete" onclick={onDelete} />
						<ActionIcon icon={Send} tooltip="Send immediately" onclick={onSendImmediately} />
					</div>
				</div>
			</div>
		</div>
	{/if}
</div>
