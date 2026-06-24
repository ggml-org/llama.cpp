<script lang="ts">
	import { ScanText, Check, X } from '@lucide/svelte';
	import { ChatMessageActionIcons, MarkdownContent } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { Card } from '$lib/components/ui/card';
	import { INPUT_CLASSES } from '$lib/constants';
	import { getMessageEditContext } from '$lib/contexts';
	import { KeyboardKey, MessageRole } from '$lib/enums';
	import { config } from '$lib/stores/settings.svelte';
	import { isIMEComposing } from '$lib/utils';

	interface Props {
		class?: string;
		message: DatabaseMessage;
		siblingInfo?: ChatMessageSiblingInfo | null;
		showDeleteDialog: boolean;
		deletionInfo: {
			totalCount: number;
			userMessages: number;
			assistantMessages: number;
			messageTypes: string[];
		} | null;
		onCopy: () => void;
		onEdit: () => void;
		onDelete: () => void;
		onConfirmDelete: () => void;
		onNavigateToSibling?: (siblingId: string) => void;
		onShowDeleteDialogChange: (show: boolean) => void;
		addToLibrary?: boolean;
		onAddToLibraryChange?: (val: boolean) => void;
		deferSystemPromptSave?: boolean;
		textareaElement?: HTMLTextAreaElement;
		instructionId?: string;
		title?: string;
		promptIsStale?: boolean;
		onPromptUpdate?: () => void;
	}

	let {
		class: className = '',
		message,
		siblingInfo = null,
		showDeleteDialog,
		deletionInfo,
		onCopy,
		onEdit,
		onDelete,
		onConfirmDelete,
		onNavigateToSibling,
		onShowDeleteDialogChange,
		addToLibrary = false,
		onAddToLibraryChange,
		deferSystemPromptSave = false,
		textareaElement = $bindable(),
		instructionId,
		title,
		promptIsStale = false,
		onPromptUpdate
	}: Props = $props();

	const editCtx = getMessageEditContext();

	function handleEditKeydown(event: KeyboardEvent) {
		if (event.key === KeyboardKey.ENTER && !event.shiftKey && !isIMEComposing(event)) {
			event.preventDefault();
			handleSave();
		} else if (event.key === KeyboardKey.ESCAPE) {
			event.preventDefault();
			handleCancel();
		}
	}

	function handleSave() {
		editCtx.save();
	}

	function handleCancel() {
		editCtx.cancel();
		onAddToLibraryChange?.(false);
	}

	let isMultiline = $state(false);
	let messageElement: HTMLElement | undefined = $state();
	let isExpanded = $state(false);
	let contentHeight = $state(0);

	const MAX_HEIGHT = 200; // pixels
	const currentConfig = config();

	let showExpandButton = $derived(contentHeight > MAX_HEIGHT);

	$effect(() => {
		if (!messageElement || !message.content.trim()) return;

		if (message.content.includes('\n')) {
			isMultiline = true;
		}

		const resizeObserver = new ResizeObserver((entries) => {
			for (const entry of entries) {
				const element = entry.target as HTMLElement;
				const estimatedSingleLineHeight = 24;

				isMultiline = element.offsetHeight > estimatedSingleLineHeight * 1.5;
				contentHeight = element.scrollHeight;
			}
		});

		resizeObserver.observe(messageElement);

		return () => {
			resizeObserver.disconnect();
		};
	});

	function toggleExpand() {
		isExpanded = !isExpanded;
	}
</script>

<div
	aria-label="System message with actions"
	class="group flex flex-col items-end gap-3 md:gap-2 {className}"
	role="group"
>
	{#if editCtx.isEditing}
		<div class="w-full max-w-[80%]">
			<textarea
				bind:this={textareaElement}
				value={editCtx.editedContent}
				class="min-h-[60px] w-full resize-none rounded-2xl px-3 py-2 text-sm {INPUT_CLASSES}"
				onkeydown={handleEditKeydown}
				oninput={(e) => editCtx.setContent(e.currentTarget.value)}
				placeholder="Edit system message..."
			></textarea>

			<div class="mt-2 flex flex-col items-end gap-2">
				{#if !deferSystemPromptSave}
					<label class="flex cursor-pointer items-center gap-2 text-sm text-muted-foreground">
						<input
							type="checkbox"
							class="h-4 w-4 rounded border-input"
							checked={addToLibrary}
							onchange={(e) => onAddToLibraryChange?.((e.target as HTMLInputElement).checked)}
						/>
						Add to library
					</label>
				{/if}

				<div class="flex gap-2">
					<Button class="h-8 px-3" onclick={handleCancel} size="sm" variant="outline">
						<X class="mr-1 h-3 w-3" />

						Cancel
					</Button>

					<Button
						class="h-8 px-3"
						onclick={handleSave}
						disabled={!editCtx.editedContent.trim()}
						size="sm"
					>
						<Check class="mr-1 h-3 w-3" />

						Save
					</Button>
				</div>
			</div>
		</div>
	{:else}
		{#if message.content.trim()}
			<div class="relative max-w-[80%]">
				<button
					class="group/expand w-full text-left {!isExpanded && showExpandButton
						? 'cursor-pointer'
						: 'cursor-auto'}"
					onclick={showExpandButton && !isExpanded ? toggleExpand : undefined}
					type="button"
				>
					<Card
						class="overflow-y-auto rounded-[1.125rem] !border-2 !border-dashed !border-border/50 bg-muted px-3.75 py-1.5 data-[multiline]:py-2.5"
						data-multiline={isMultiline ? '' : undefined}
						style="border: 2px dashed hsl(var(--border)); max-height: var(--max-message-height); overflow-wrap: anywhere; word-break: break-word;"
					>
						<div
							class="relative transition-all duration-300 {isExpanded
								? 'cursor-text select-text'
								: 'select-none'}"
							style={!isExpanded && showExpandButton
								? `max-height: ${MAX_HEIGHT}px;`
								: 'max-height: none;'}
						>
							{#if currentConfig.renderUserContentAsMarkdown}
								<div bind:this={messageElement} class={isExpanded ? 'cursor-text' : ''}>
									<MarkdownContent
										class="markdown-system-content -my-4"
										content={message.content}
									/>
								</div>
							{:else}
								<span
									bind:this={messageElement}
									class="text-md whitespace-pre-wrap {isExpanded ? 'cursor-text' : ''}"
								>
									{message.content}
								</span>
							{/if}

							{#if !isExpanded && showExpandButton}
								<div
									class="pointer-events-none absolute right-0 bottom-0 left-0 h-48 bg-gradient-to-t from-muted to-transparent"
								></div>

								<div
									class="pointer-events-none absolute right-0 bottom-4 left-0 flex justify-center opacity-0 transition-opacity group-hover/expand:opacity-100"
								>
									<Button
										class="rounded-full px-4 py-1.5 text-xs shadow-md"
										size="sm"
										variant="outline"
									>
										Show full system message
									</Button>
								</div>
							{/if}
						</div>

						{#if isExpanded && showExpandButton}
							<div class="mb-2 flex justify-center">
								<Button
									class="rounded-full px-4 py-1.5 text-xs"
									onclick={(e) => {
										e.stopPropagation();
										toggleExpand();
									}}
									size="sm"
									variant="outline"
								>
									Collapse System Message
								</Button>
							</div>
						{/if}
					</Card>
				</button>
			</div>
		{/if}

		{#if instructionId && title}
			<div class="max-w-[80%] flex items-center gap-2">
				<ScanText class="h-3.5 w-3.5 text-muted-foreground" />

				<span class="text-xs font-medium text-muted-foreground">{title}</span>

				{#if promptIsStale}
					<button
						type="button"
						class="text-xs font-medium text-amber-600 hover:underline hover:text-amber-700 dark:text-amber-400 dark:hover:text-amber-300"
						onclick={onPromptUpdate}
					>
						Newer version available
					</button>
				{/if}
			</div>
		{/if}

		{#if message.timestamp}
			<div class="max-w-[80%]">
				<ChatMessageActionIcons
					actionsPosition="right"
					{deletionInfo}
					justify="end"
					{onConfirmDelete}
					{onCopy}
					{onDelete}
					{onEdit}
					{onNavigateToSibling}
					{onShowDeleteDialogChange}
					{siblingInfo}
					{showDeleteDialog}
					role={MessageRole.USER}
				/>
			</div>
		{/if}
	{/if}
</div>
