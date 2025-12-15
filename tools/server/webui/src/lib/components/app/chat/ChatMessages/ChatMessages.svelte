<script lang="ts">
	import { ChatMessage } from '$lib/components/app';
	import { setChatActionsContext } from '$lib/contexts';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { conversationsStore, activeConversation } from '$lib/stores/conversations.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { copyToClipboard, formatMessageForClipboard, getMessageSiblings } from '$lib/utils';
	import { SvelteSet } from 'svelte/reactivity';
	import type {
		ApiChatCompletionToolCall,
		ChatMessageSiblingInfo,
		ChatMessageTimings,
		DatabaseMessage
	} from '$lib/types';

	interface Props {
		class?: string;
		messages?: DatabaseMessage[];
		onUserAction?: () => void;
	}

	let { class: className, messages = [], onUserAction }: Props = $props();

	// Prefer live store messages; fall back to the provided prop (e.g. initial render/tests).
	const sourceMessages = $derived(
		conversationsStore.activeMessages.length ? conversationsStore.activeMessages : messages
	);

	let allConversationMessages = $state<DatabaseMessage[]>([]);
	const currentConfig = config();

	setChatActionsContext({
		copy: async (message: DatabaseMessage) => {
			const asPlainText = Boolean(currentConfig.copyTextAttachmentsAsPlainText);
			const clipboardContent = formatMessageForClipboard(
				message.content,
				message.extra,
				asPlainText
			);
			await copyToClipboard(clipboardContent, 'Message copied to clipboard');
		},
		delete: async (message: DatabaseMessage) => {
			await handleDeleteMessage(message);
		},
		navigateToSibling: async (siblingId: string) => {
			await handleNavigateToSibling(siblingId);
		},
		editWithBranching: async (
			message: DatabaseMessage,
			newContent: string,
			newExtras?: DatabaseMessageExtra[]
		) => {
			await handleEditWithBranching(message, newContent, newExtras);
		},
		editWithReplacement: async (
			message: DatabaseMessage,
			newContent: string,
			shouldBranch: boolean
		) => {
			await handleEditWithReplacement(message, newContent, shouldBranch);
		},
		editUserMessagePreserveResponses: async (
			message: DatabaseMessage,
			newContent: string,
			newExtras?: DatabaseMessageExtra[]
		) => {
			await handleEditUserMessagePreserveResponses(message, newContent, newExtras);
		},
		regenerateWithBranching: async (message: DatabaseMessage, modelOverride?: string) => {
			await handleRegenerateWithBranching(message, modelOverride);
		},
		continueAssistantMessage: async (message: DatabaseMessage) => {
			await handleContinueAssistantMessage(message);
		}
	});

	function refreshAllMessages() {
		const conversation = activeConversation();

		if (conversation) {
			conversationsStore.getConversationMessages(conversation.id).then((messages) => {
				allConversationMessages = messages;
			});
		} else {
			allConversationMessages = [];
		}
	}

	// Single effect that tracks both conversation and message changes
	$effect(() => {
		const conversation = activeConversation();

		if (conversation) {
			refreshAllMessages();
		}
	});

	type ToolSegment =
		| { kind: 'content'; content: string; parentId: string }
		| { kind: 'thinking'; content: string }
		| {
				kind: 'tool';
				toolCalls: ApiChatCompletionToolCall[];
				parentId: string;
				inThinking: boolean;
		  };
	type CollectedToolMessage = {
		toolCallId?: string | null;
		parsed: { expression?: string; result?: string; duration_ms?: number };
	};
	type AssistantDisplayMessage = DatabaseMessage & {
		_toolParentIds?: string[];
		_segments?: ToolSegment[];
		_toolMessagesCollected?: CollectedToolMessage[];
		_actionTargetId?: string;
	};
	type DisplayEntry = {
		message: DatabaseMessage | AssistantDisplayMessage;
		siblingInfo: ChatMessageSiblingInfo;
	};

	let displayMessages = $derived.by((): DisplayEntry[] => {
		// Force reactivity on message field changes (important for streaming updates)
		const signature = sourceMessages
			.map(
				(m) =>
					`${m.id}-${m.role}-${m.parent ?? ''}-${m.timestamp ?? ''}-${m.thinking ?? ''}-${
						m.toolCalls ?? ''
					}-${m.content ?? ''}`
			)
			.join('|');
		// signature is unused but ensures Svelte tracks the above fields
		void signature;

		if (!sourceMessages.length) return [];

		// Filter out system messages if showSystemMessage is false
		const filteredMessages = currentConfig.showSystemMessage
			? sourceMessages
			: sourceMessages.filter((msg) => msg.type !== 'system');
		const sortedFilteredMessages = [...filteredMessages].sort(
			(a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0) || a.id.localeCompare(b.id)
		);

		const visited = new SvelteSet<string>();
		const result: DisplayEntry[] = [];

		const getChildren = (parentId: string, role?: string) =>
			sortedFilteredMessages
				.filter((m) => m.parent === parentId && (!role || m.role === role))
				.sort((a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0) || a.id.localeCompare(b.id));

		const normalizeToolParsed = (
			value: unknown
		): { expression?: string; result?: string; duration_ms?: number } | null => {
			if (!value || typeof value !== 'object') return null;
			const obj = value as Record<string, unknown>;
			return {
				expression: typeof obj.expression === 'string' ? obj.expression : undefined,
				result: typeof obj.result === 'string' ? obj.result : undefined,
				duration_ms: typeof obj.duration_ms === 'number' ? obj.duration_ms : undefined
			};
		};

		const sumTimings = (assistantIds: string[]): ChatMessageTimings | undefined => {
			if (assistantIds.length <= 1) return undefined;

			let predicted_n_sum = 0;
			let predicted_ms_sum = 0;
			let prompt_n_sum = 0;
			let prompt_ms_sum = 0;
			let cache_n_sum = 0;

			let hasPredicted = false;
			let hasPrompt = false;
			let hasCache = false;

			for (const id of assistantIds) {
				const m = filteredMessages.find((x) => x.id === id);
				const t = m?.timings;
				if (!t) continue;

				if (typeof t.predicted_n === 'number' && typeof t.predicted_ms === 'number') {
					predicted_n_sum += t.predicted_n;
					predicted_ms_sum += t.predicted_ms;
					hasPredicted = true;
				}

				if (typeof t.prompt_n === 'number' && typeof t.prompt_ms === 'number') {
					prompt_n_sum += t.prompt_n;
					prompt_ms_sum += t.prompt_ms;
					hasPrompt = true;
				}

				if (typeof t.cache_n === 'number') {
					cache_n_sum += t.cache_n;
					hasCache = true;
				}
			}

			if (!hasPredicted && !hasPrompt && !hasCache) return undefined;

			return {
				...(hasPredicted ? { predicted_n: predicted_n_sum, predicted_ms: predicted_ms_sum } : {}),
				...(hasPrompt ? { prompt_n: prompt_n_sum, prompt_ms: prompt_ms_sum } : {}),
				...(hasCache ? { cache_n: cache_n_sum } : {})
			};
		};

		for (const msg of sortedFilteredMessages) {
			if (visited.has(msg.id)) continue;
			// Don't render tools directly, but keep them for collection; skip marking visited here

			// Skip tool messages (rendered inline)
			if (msg.role === 'tool') continue;

			if (msg.role === 'assistant') {
				// Collapse consecutive assistant/tool chains into one display message
				const toolParentIds: string[] = [];
				const thinkingParts: string[] = [];
				const contentParts: string[] = [];
				const toolCallsCombined: ApiChatCompletionToolCall[] = [];
				const segments: ToolSegment[] = [];
				const toolMessagesCollected: CollectedToolMessage[] = [];
				const toolCallIds = new SvelteSet<string>();

				let currentAssistant: DatabaseMessage | undefined = msg;

				while (currentAssistant) {
					visited.add(currentAssistant.id);
					toolParentIds.push(currentAssistant.id);

					if (currentAssistant.thinking) {
						thinkingParts.push(currentAssistant.thinking);
						segments.push({ kind: 'thinking', content: currentAssistant.thinking });
					}

					const hasContent = Boolean(currentAssistant.content?.trim());
					if (hasContent) {
						contentParts.push(currentAssistant.content);
						segments.push({
							kind: 'content',
							content: currentAssistant.content,
							parentId: currentAssistant.id
						});
					}
					let thisAssistantToolCalls: ApiChatCompletionToolCall[] = [];
					if (currentAssistant.toolCalls) {
						try {
							const parsed: unknown = JSON.parse(currentAssistant.toolCalls);
							if (Array.isArray(parsed)) {
								for (const tc of parsed as ApiChatCompletionToolCall[]) {
									if (tc?.id && toolCallIds.has(tc.id)) continue;
									if (tc?.id) toolCallIds.add(tc.id);
									toolCallsCombined.push(tc);
									thisAssistantToolCalls.push(tc);
								}
							}
						} catch {
							// ignore malformed
						}
					}
					if (thisAssistantToolCalls.length) {
						const toolCallsInThinking = Boolean(currentAssistant.thinking) && !hasContent;
						segments.push({
							kind: 'tool',
							toolCalls: thisAssistantToolCalls,
							parentId: currentAssistant.id,
							// Treat tool calls as in-reasoning only when this assistant step is still thought-only.
							inThinking: toolCallsInThinking
						});
					}

					const directToolChildren = getChildren(currentAssistant.id, 'tool');
					const toolChildren: DatabaseMessage[] = [];
					const toolQueue = [...directToolChildren];
					const seenToolIds = new SvelteSet<string>();
					let nextAssistantFromToolChain: DatabaseMessage | undefined;

					while (toolQueue.length > 0) {
						const toolMsg = toolQueue.shift();
						if (!toolMsg || seenToolIds.has(toolMsg.id)) continue;
						seenToolIds.add(toolMsg.id);
						toolChildren.push(toolMsg);

						const nestedTools = getChildren(toolMsg.id, 'tool');
						if (nestedTools.length) {
							toolQueue.push(...nestedTools);
						}

						const assistantChildren = getChildren(toolMsg.id, 'assistant');
						const candidate = assistantChildren[0];
						if (candidate) {
							const candidateKey = `${candidate.timestamp ?? 0}-${candidate.id}`;
							const currentKey = nextAssistantFromToolChain
								? `${nextAssistantFromToolChain.timestamp ?? 0}-${nextAssistantFromToolChain.id}`
								: '';
							if (!nextAssistantFromToolChain || candidateKey < currentKey) {
								nextAssistantFromToolChain = candidate;
							}
						}
					}

					for (const t of toolChildren) {
						visited.add(t.id);
						// capture parsed tool message for inline use
						try {
							const parsedUnknown: unknown = t.content ? JSON.parse(t.content) : null;
							const normalized = normalizeToolParsed(parsedUnknown);
							toolMessagesCollected.push({
								toolCallId: t.toolCallId,
								parsed: normalized ?? { result: t.content }
							});
						} catch {
							const p = { result: t.content };
							toolMessagesCollected.push({ toolCallId: t.toolCallId, parsed: p });
						}
					}

					// Follow continuation assistant from the collected tool chain first.
					let nextAssistant = nextAssistantFromToolChain;

					// Also allow direct assistant->assistant continuation (no intervening tool)
					if (!nextAssistant) {
						nextAssistant = getChildren(currentAssistant.id, 'assistant')[0];
					}

					if (nextAssistant) {
						currentAssistant = nextAssistant;
						continue;
					}
					break;
				}

				const siblingInfo: ChatMessageSiblingInfo = getMessageSiblings(
					allConversationMessages,
					msg.id
				) || {
					message: msg,
					siblingIds: [msg.id],
					currentIndex: 0,
					totalSiblings: 1
				};

				const aggregatedTimings = sumTimings(toolParentIds);

				const mergedAssistant: AssistantDisplayMessage = {
					...(currentAssistant ?? msg),
					// Keep a plain-text combined content for edit/copy; display can use `_segments` for ordering.
					content: contentParts.filter(Boolean).join('\n\n'),
					thinking: thinkingParts.filter(Boolean).join('\n\n'),
					toolCalls: toolCallsCombined.length ? JSON.stringify(toolCallsCombined) : '',
					...(aggregatedTimings ? { timings: aggregatedTimings } : {}),
					_toolParentIds: toolParentIds,
					_segments: segments,
					_actionTargetId: msg.id,
					_toolMessagesCollected: toolMessagesCollected
				};

				result.push({ message: mergedAssistant, siblingInfo });
				continue;
			}

			// user/system messages
			const siblingInfo: ChatMessageSiblingInfo = getMessageSiblings(
				allConversationMessages,
				msg.id
			) || {
				message: msg,
				siblingIds: [msg.id],
				currentIndex: 0,
				totalSiblings: 1
			};
			result.push({ message: msg, siblingInfo });
		}

		return result;
	});

	function getToolParentIdsForMessage(msg: DisplayEntry['message']): string[] | undefined {
		return (msg as AssistantDisplayMessage)._toolParentIds;
	}

	function getDisplayKeyForMessage(msg: DisplayEntry['message']): string {
		return (msg as AssistantDisplayMessage)._actionTargetId ?? msg.id;
	}

	async function handleNavigateToSibling(siblingId: string) {
		await conversationsStore.navigateToSibling(siblingId);
	}

	async function handleEditWithBranching(
		message: DatabaseMessage,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	) {
		onUserAction?.();

		await chatStore.editMessageWithBranching(message.id, newContent, newExtras);

		refreshAllMessages();
	}

	async function handleEditWithReplacement(
		message: DatabaseMessage,
		newContent: string,
		shouldBranch: boolean
	) {
		onUserAction?.();

		await chatStore.editAssistantMessage(message.id, newContent, shouldBranch);

		refreshAllMessages();
	}

	async function handleRegenerateWithBranching(message: DatabaseMessage, modelOverride?: string) {
		onUserAction?.();

		await chatStore.regenerateMessageWithBranching(message.id, modelOverride);

		refreshAllMessages();
	}

	async function handleContinueAssistantMessage(message: DatabaseMessage) {
		onUserAction?.();

		await chatStore.continueAssistantMessage(message.id);

		refreshAllMessages();
	}

	async function handleEditUserMessagePreserveResponses(
		message: DatabaseMessage,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	) {
		onUserAction?.();

		await chatStore.editUserMessagePreserveResponses(message.id, newContent, newExtras);

		refreshAllMessages();
	}

	async function handleDeleteMessage(message: DatabaseMessage) {
		await chatStore.deleteMessage(message.id);

		refreshAllMessages();
	}
</script>

<div class="flex h-full flex-col space-y-10 pt-16 md:pt-24 {className}" style="height: auto; ">
	{#each displayMessages as { message, siblingInfo } (getDisplayKeyForMessage(message))}
		<ChatMessage
			class="mx-auto w-full max-w-[48rem]"
			{message}
			{siblingInfo}
			toolParentIds={getToolParentIdsForMessage(message)}
		/>
	{/each}
</div>
