<script lang="ts">
	import type { HfShareScope, HfTraceFile } from '@ggml-org/llama-huggingface';
	import { HfShareDialog } from '@ggml-org/llama-huggingface/svelte';
	import { CONVERSATION_SHARE_TEXT as TEXT, HUGGING_FACE_ENABLED } from '#lib/constants/index.ts';
	import { ConversationShareScope } from '#lib/enums/index.ts';
	import { ConversationTransferService } from '#lib/services/conversation-transfer.service.ts';
	import { conversationsStore, huggingFace, uiStore } from '#lib/stores/index.ts';
	import {
		buildConversationTrace,
		buildTracePreview,
		copyToClipboard,
		filterByLeafNodeId
	} from '#lib/utils/index.ts';
	import { toast } from 'svelte-sonner';

	// what a conversation is, for the share dialog: its branches as scopes, and its trace

	let options = $derived(uiStore.conversationShare);
	let conversation = $derived(
		conversationsStore.conversations.find((conv) => conv.id === options?.conversationId)
	);

	let messages = $state<DatabaseMessage[]>([]);

	$effect.pre(() => {
		const conversationId = options?.conversationId;

		messages = [];

		if (!conversationId) return;

		void conversationsStore.getConversationMessages(conversationId).then((loaded) => {
			if (uiStore.conversationShare?.conversationId === conversationId) messages = loaded;
		});
	});

	function scopeOf(id: ConversationShareScope, label: string, leafId: string): HfShareScope {
		const path = filterByLeafNodeId(messages, leafId);

		return {
			// nothing to count while the messages load
			detail: path.length ? TEXT.messageCount(path.length) : undefined,
			hasAttachments: path.some((message) => !!message.extra?.length),
			id,
			isEmpty: !path.length,
			label
		};
	}

	let scopes = $derived([
		...(options?.messageId
			? [scopeOf(ConversationShareScope.UP_TO_MESSAGE, TEXT.SCOPE_UP_TO_MESSAGE, options.messageId)]
			: []),
		scopeOf(ConversationShareScope.ALL, TEXT.SCOPE_ALL, conversation?.currNode ?? '')
	]);

	async function buildTrace(scope: string, includeAttachments: boolean): Promise<HfTraceFile> {
		const conversationId = options?.conversationId ?? '';
		const [exported] = await conversationsStore.getConversationsForExport([conversationId]);

		if (!exported) throw new Error(TEXT.CONVERSATION_NOT_FOUND);

		const trace = buildConversationTrace(exported, {
			includeAttachments,
			upToMessageId: scope === ConversationShareScope.UP_TO_MESSAGE ? options?.messageId : undefined
		});

		return {
			commitMessage: TEXT.commitMessage(trace.conv.name),
			content: ConversationTransferService.serializeSessionToJsonl(trace),
			path: ConversationTransferService.generateConversationFilename(trace.conv, trace.messages),
			preview: buildTracePreview(trace)
		};
	}
</script>

{#if HUGGING_FACE_ENABLED}
	<HfShareDialog
		bind:open={() => options !== null, (open) => !open && (uiStore.conversationShare = null)}
		account={huggingFace}
		{buildTrace}
		onCopyLink={(url) => void copyToClipboard(url, TEXT.LINK_COPIED)}
		onShared={() => toast.success(TEXT.SHARED)}
		{scopes}
		subject={conversation ? TEXT.subject(conversation.name) : TEXT.SUBJECT_FALLBACK}
	/>
{/if}
