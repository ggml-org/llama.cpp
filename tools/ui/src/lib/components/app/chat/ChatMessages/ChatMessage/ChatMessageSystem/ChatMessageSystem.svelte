<script lang="ts">
	import { ScanText } from '@lucide/svelte';
	import {
		ChatMessageActionIcons,
		ChatMessageEditForm,
		MarkdownContent
	} from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { Card } from '$lib/components/ui/card';
	import { getMessageEditContext } from '$lib/contexts';
	import { ChatMessageEditFormVariant, MessageRole } from '$lib/enums';
	import { config } from '$lib/stores/settings.svelte';

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
		skillId?: string;
		title?: string;
		skillIsStale?: boolean;
		onSkillUpdate?: () => void;
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
		skillId,
		title,
		skillIsStale = false,
		onSkillUpdate
	}: Props = $props();

	const editCtx = getMessageEditContext();

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
		<div class="flex w-full flex-col items-end gap-2">
			<ChatMessageEditForm variant={ChatMessageEditFormVariant.SYSTEM} />
		</div>
	{:else}
		{#if message.content.trim()}
			<div class="relative">
				<button
					class="group/expand w-full text-left {!isExpanded && showExpandButton
						? 'cursor-pointer'
						: 'cursor-auto'}"
					onclick={showExpandButton && !isExpanded ? toggleExpand : undefined}
					type="button"
				>
					<Card
						class="overflow-y-auto rounded-[1.125rem] border-2! border-dashed! border-border/50! bg-muted px-3.75 py-1.5 data-multiline:py-2.5"
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

		{#if skillId && title}
			<div class="flex items-center gap-2">
				<ScanText class="h-3.5 w-3.5 text-muted-foreground" />

				<span class="text-xs font-medium text-muted-foreground">{title}</span>

				{#if skillIsStale}
					<button
						type="button"
						class="text-xs font-medium text-amber-600 hover:underline hover:text-amber-700 dark:text-amber-400 dark:hover:text-amber-300"
						onclick={onSkillUpdate}
						title="This skill has been modified. Click to save the changes back to the library."
					>
						Modified
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
