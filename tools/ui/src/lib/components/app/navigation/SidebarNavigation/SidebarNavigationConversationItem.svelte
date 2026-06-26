<script lang="ts">
	import {
		Trash2,
		Pencil,
		MoreHorizontal,
		Download,
		Loader2,
		Square,
		GitBranch,
		Pin,
		PinOff
	} from '@lucide/svelte';
	import { DropdownMenuActions } from '$lib/components/app';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { FORK_TREE_DEPTH_PADDING } from '$lib/constants';
	import { RouterService } from '$lib/services/router.service';
	import { getAllLoadingChats } from '$lib/stores/chat.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { TruncatedText } from '$lib/components/app';
	import { onMount } from 'svelte';

	interface Props {
		isActive?: boolean;
		depth?: number;
		conversation: DatabaseConversation;
		onDelete?: (id: string) => void;
		onEdit?: (id: string) => void;
		onSelect?: (id: string) => void;
		onStop?: (id: string) => void;
	}

	let {
		conversation,
		onDelete,
		onEdit,
		onSelect,
		onStop,
		isActive = false,
		depth = 0
	}: Props = $props();

	let dropdownOpen = $state(false);

	let isLoading = $derived(getAllLoadingChats().includes(conversation.id));

	function handleEdit(event: Event) {
		event.stopPropagation();
		onEdit?.(conversation.id);
	}

	function handleDelete(event: Event) {
		event.stopPropagation();
		onDelete?.(conversation.id);
	}

	function handleStop(event: Event) {
		event.stopPropagation();
		onStop?.(conversation.id);
	}

	function handleTogglePin() {
		conversationsStore.toggleConversationPin(conversation.id);
	}

	function handleGlobalEditEvent(event: Event) {
		const customEvent = event as CustomEvent<{ conversationId: string }>;

		if (customEvent.detail.conversationId === conversation.id && isActive) {
			handleEdit(event);
		}
	}

	function handleSelect() {
		onSelect?.(conversation.id);
	}

	onMount(() => {
		document.addEventListener('edit-active-conversation', handleGlobalEditEvent as EventListener);

		return () => {
			document.removeEventListener(
				'edit-active-conversation',
				handleGlobalEditEvent as EventListener
			);
		};
	});
</script>

<div
	class="conversation-item group relative flex min-h-9 w-full items-center justify-between space-x-3 rounded-lg py-1.5 transition-colors hover:bg-foreground/10 {isActive
		? 'bg-foreground/5 text-accent-foreground'
		: ''} px-3"
>
	<button
		class="absolute inset-0 z-0 cursor-pointer rounded-lg focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
		onclick={handleSelect}
		aria-label={conversation.name}
	>
	</button>
	<div
		class="pointer-events-none relative z-10 flex min-w-0 flex-1 items-center gap-2"
		style:padding-left="{depth * FORK_TREE_DEPTH_PADDING}px"
	>
		{#if depth > 0}
			<Tooltip.Root>
				<Tooltip.Trigger>
					<!-- prevent another nested button element -->
					{#snippet child({ props })}
						<a
							{...props}
							href={RouterService.chat(conversation.forkedFromConversationId)}
							class="pointer-events-auto flex shrink-0 items-center text-muted-foreground transition-colors hover:text-foreground"
						>
							<GitBranch class="h-3.5 w-3.5" />
						</a>
					{/snippet}
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>See parent conversation</p>
				</Tooltip.Content>
			</Tooltip.Root>
		{/if}

		{#if isLoading}
			<Tooltip.Root>
				<Tooltip.Trigger>
					<button
						class="stop-button pointer-events-auto flex h-4 w-4 shrink-0 cursor-pointer items-center justify-center rounded text-muted-foreground transition-colors hover:text-foreground"
						onclick={handleStop}
						aria-label="Stop generation"
					>
						<Loader2 class="loading-icon h-3.5 w-3.5 animate-spin" />

						<Square class="stop-icon hidden h-3 w-3 fill-current text-destructive" />
					</button>
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>Stop generation</p>
				</Tooltip.Content>
			</Tooltip.Root>
		{/if}

		<TruncatedText text={conversation.name} class="text-sm font-medium" showTooltip={false} />
	</div>

	<div class="actions pointer-events-auto relative z-20 flex items-center">
		<DropdownMenuActions
			triggerIcon={MoreHorizontal}
			triggerTooltip="More actions"
			bind:open={dropdownOpen}
			actions={[
				{
					icon: conversation.pinned ? PinOff : Pin,
					label: conversation.pinned ? 'Unpin' : 'Pin',
					onclick: (e: Event) => {
						e.stopPropagation();
						handleTogglePin();
					}
				},
				{
					icon: Pencil,
					label: 'Edit',
					onclick: handleEdit,
					shortcut: ['shift', 'cmd', 'e']
				},
				{
					icon: Download,
					label: 'Export',
					onclick: (e: Event) => {
						e.stopPropagation();
						conversationsStore.downloadConversation(conversation.id);
					},
					shortcut: ['shift', 'cmd', 's']
				},
				{
					icon: Trash2,
					label: 'Delete',
					onclick: handleDelete,
					variant: 'destructive',
					shortcut: ['shift', 'cmd', 'd'],
					separator: true
				}
			]}
		/>
	</div>
</div>

<style>
	.conversation-item {
		:global([data-slot='dropdown-menu-trigger']:not([data-state='open'])) {
			opacity: 0;
		}

		&:is(:hover) :global([data-slot='dropdown-menu-trigger']),
		&:focus-within :global([data-slot='dropdown-menu-trigger']) {
			opacity: 1;
		}
		@media (max-width: 768px) {
			:global([data-slot='dropdown-menu-trigger']) {
				opacity: 1 !important;
			}
		}

		.stop-button {
			:global(.stop-icon) {
				display: none;
			}

			:global(.loading-icon) {
				display: block;
			}
		}

		&:is(:hover) .stop-button,
		&:focus-within .stop-button {
			:global(.stop-icon) {
				display: block;
			}

			:global(.loading-icon) {
				display: none;
			}
		}
	}
</style>
