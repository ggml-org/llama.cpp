<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { ConversationSelection } from '$lib/components/app';
	import { t } from '$lib/i18n';

	interface Props {
		conversations: DatabaseConversation[];
		messageCountMap?: Map<string, number>;
		mode: 'export' | 'import';
		onCancel: () => void;
		onConfirm: (selectedConversations: DatabaseConversation[]) => void;
		open?: boolean;
	}

	let {
		conversations,
		messageCountMap = new Map(),
		mode,
		onCancel,
		onConfirm,
		open = $bindable(false)
	}: Props = $props();

	let conversationSelectionRef: ConversationSelection | undefined = $state();

	let previousOpen = $state(false);

	$effect(() => {
		if (open && !previousOpen && conversationSelectionRef) {
			conversationSelectionRef.reset();
		} else if (!open && previousOpen) {
			onCancel();
		}

		previousOpen = open;
	});

	let actionLabel = $derived(
		mode === 'export'
			? t('dialog.conversation_selection.action_export')
			: t('dialog.conversation_selection.action_import')
	);
</script>

<Dialog.Root bind:open>
	<Dialog.Portal>
		<Dialog.Overlay class="z-[1000000]" />

		<Dialog.Content class="z-[1000001] max-w-2xl">
			<Dialog.Header>
				<Dialog.Title>
					{t('dialog.conversation_selection.title', { action: actionLabel })}
				</Dialog.Title>
				<Dialog.Description>
					{#if mode === 'export'}
						{t('dialog.conversation_selection.description_export')}
					{:else}
						{t('dialog.conversation_selection.description_import')}
					{/if}
				</Dialog.Description>
			</Dialog.Header>

			<ConversationSelection
				bind:this={conversationSelectionRef}
				{conversations}
				{messageCountMap}
				{mode}
				{onCancel}
				{onConfirm}
			/>
		</Dialog.Content>
	</Dialog.Portal>
</Dialog.Root>
