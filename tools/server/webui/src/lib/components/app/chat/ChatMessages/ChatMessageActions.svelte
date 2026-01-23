<script lang="ts">
	import { Edit, Copy, RefreshCw, Trash2, ArrowRight } from '@lucide/svelte';
	import {
		ActionButton,
		ChatMessageBranchingControls,
		DialogConfirmation
	} from '$lib/components/app';
	import { t } from '$lib/i18n';

	interface Props {
		role: 'user' | 'assistant';
		justify: 'start' | 'end';
		actionsPosition: 'left' | 'right';
		siblingInfo?: ChatMessageSiblingInfo | null;
		showDeleteDialog: boolean;
		deletionInfo: {
			totalCount: number;
			userMessages: number;
			assistantMessages: number;
			messageTypes: string[];
		} | null;
		onCopy: () => void;
		onEdit?: () => void;
		onRegenerate?: () => void;
		onContinue?: () => void;
		onDelete: () => void;
		onConfirmDelete: () => void;
		onNavigateToSibling?: (siblingId: string) => void;
		onShowDeleteDialogChange: (show: boolean) => void;
	}

	let {
		actionsPosition,
		deletionInfo,
		justify,
		onCopy,
		onEdit,
		onConfirmDelete,
		onContinue,
		onDelete,
		onNavigateToSibling,
		onShowDeleteDialogChange,
		onRegenerate,
		role,
		siblingInfo = null,
		showDeleteDialog
	}: Props = $props();

	function handleConfirmDelete() {
		onConfirmDelete();
		onShowDeleteDialogChange(false);
	}
</script>

<div class="relative {justify === 'start' ? 'mt-2' : ''} flex h-6 items-center justify-{justify}">
	<div
		class="absolute top-0 {actionsPosition === 'left'
			? 'left-0'
			: 'right-0'} flex items-center gap-2 opacity-100 transition-opacity"
	>
		{#if siblingInfo && siblingInfo.totalSiblings > 1}
			<ChatMessageBranchingControls {siblingInfo} {onNavigateToSibling} />
		{/if}

		<div
			class="pointer-events-auto inset-0 flex items-center gap-1 opacity-100 transition-all duration-150"
		>
			<ActionButton icon={Copy} tooltip={t('chat.message.actions.copy')} onclick={onCopy} />

			{#if onEdit}
				<ActionButton icon={Edit} tooltip={t('chat.message.actions.edit')} onclick={onEdit} />
			{/if}

			{#if role === 'assistant' && onRegenerate}
				<ActionButton
					icon={RefreshCw}
					tooltip={t('chat.message.actions.regenerate')}
					onclick={() => onRegenerate()}
				/>
			{/if}

			{#if role === 'assistant' && onContinue}
				<ActionButton
					icon={ArrowRight}
					tooltip={t('chat.message.actions.continue')}
					onclick={onContinue}
				/>
			{/if}

			<ActionButton icon={Trash2} tooltip={t('chat.message.actions.delete')} onclick={onDelete} />
		</div>
	</div>
</div>

<DialogConfirmation
	bind:open={showDeleteDialog}
	title={t('chat.message.delete.title')}
	description={deletionInfo && deletionInfo.totalCount > 1
		? t('chat.message.delete.description_many', {
				total: deletionInfo.totalCount,
				userCount: deletionInfo.userMessages,
				userLabel:
					deletionInfo.userMessages === 1
						? t('chat.message.delete.user_singular')
						: t('chat.message.delete.user_plural'),
				assistantCount: deletionInfo.assistantMessages,
				assistantLabel:
					deletionInfo.assistantMessages === 1
						? t('chat.message.delete.assistant_singular')
						: t('chat.message.delete.assistant_plural')
			})
		: t('chat.message.delete.description_single')}
	confirmText={deletionInfo && deletionInfo.totalCount > 1
		? t('chat.message.delete.confirm_many', { count: deletionInfo.totalCount })
		: t('chat.message.delete.confirm_single')}
	cancelText={t('chat.message.delete.cancel')}
	variant="destructive"
	icon={Trash2}
	onConfirm={handleConfirmDelete}
	onCancel={() => onShowDeleteDialogChange(false)}
/>
