/**
 * ChatStreamManager - Server-side stream sessions for conversations
 *
 * Owns the attach lifecycle for streams that live on the server: discovery,
 * replay from byte 0, and resume retry while the owning model loads. The
 * remote-running snapshot it produces feeds the chat activity ledger
 * (chatStore.activity), which owns the actual running-conv state. Created
 * and owned by chatStore; the host exposes the per-conversation state setters.
 */

import { CONVERSATION_ID_SEPARATOR, STREAM_RESUME_RETRY_MS } from '$lib/constants';
import { MessageRole, MessageType, StreamConnectionState } from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { DatabaseService } from '$lib/services/database.service';
import type { ChatActivityStore } from '$lib/stores/chat/activity.svelte';
import type { ChatProcessingStore } from '$lib/stores/chat/processing.svelte';
// direct imports between stores, not via the barrel, to avoid circular deps
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import { modelsStore } from '$lib/stores/models/index.svelte';
import type { ApiStreamSession, ChatMessageTimings, DatabaseMessage } from '$lib/types';
import { streamIdentity } from '$lib/utils';
import { SvelteMap, SvelteSet } from 'svelte/reactivity';

/**
 * The slice of chatStore the manager drives. Kept narrow on purpose so the
 * manager cannot reach around the host's full surface; chatStore implements
 * this structurally.
 */
export interface ChatStreamHost {
	activity: ChatActivityStore;
	processing: ChatProcessingStore;
	chatStreamingStates: SvelteMap<
		string,
		{ response: string; messageId: string; model?: string | null }
	>;
	streamConnectionState: StreamConnectionState;
	getOrCreateAbortController(convId: string): AbortController;
	setChatLoading(convId: string, loading: boolean): void;
	setChatStreaming(
		convId: string,
		response: string,
		messageId: string,
		model?: string | null
	): void;
	clearChatStreaming(convId: string, messageId?: string): void;
}

export class ChatStreamManager {
	// in-flight discoverActiveStream guard, keyed by conv id
	private discoveringConvs = new SvelteSet<string>();
	// convs whose resume waits on a model load: their loading state belongs to the retry loop,
	// so discoverActiveStream must not treat it as a live send and bail
	private resumePendingConvs = new SvelteSet<string>();
	// pending resume retry timers while an owning model loads, one per conv
	private resumeRetryTimers = new SvelteMap<string, ReturnType<typeof setTimeout>>();

	/** Kill a pending resume retry, e.g. on explicit stop. */
	cancelResumeRetry(convId: string): void {
		const timer = this.resumeRetryTimers.get(convId);

		if (timer !== undefined) {
			clearTimeout(timer);
			this.resumeRetryTimers.delete(convId);
		}

		this.resumePendingConvs.delete(convId);
	}

	constructor(private host: ChatStreamHost) {}

	async discoverActiveStream(convId: string): Promise<void> {
		if (!convId) return;

		if (this.host.chatStreamingStates.has(convId)) return;

		if (this.host.activity.isLocal(convId) && !this.resumePendingConvs.has(convId)) return;

		// concurrency guard: another discover may already be running for this conv (typical race
		// between mount and visibilitychange on tab switch). a second concurrent fetch on the same
		// /v1/stream would duplicate every byte into the DB message, this guard bounces it
		if (this.discoveringConvs.has(convId)) return;

		this.discoveringConvs.add(convId);

		try {
			// the model is frozen at POST time, rebuild the exact conv::model identity from the
			// persisted state so the lookup key matches what the server stored. null means a single
			// model conv with no ::suffix, only guess from the dropdown with no persisted state
			const localState = ChatService.getStreamState(convId);
			const streamId = ChatService.resumeStreamIdentity(
				convId,
				localState,
				modelsStore.selectedModelName
			);
			// primary path: ask the server which sessions exist for this identity
			const serverTarget = await this.probeServerStream(streamId);

			if (serverTarget) {
				// pass the full server side identity (may carry a ::model suffix) so the GET routes
				// straight to the owning session, no probe or fan out
				await this.attachServerStream(convId, serverTarget.conversation_id);

				return;
			}

			// fallback: local state remembers an interrupted byte offset for this conv, the server may
			// still have a live session matching that identity (we just lost the bytes mid stream). retry
			// with the frozen identity, the server probe inside attachServerStream tells us if it exists
			if (!localState) {
				return;
			}

			// quiet status probe first: a full attach flips the loading UI on every try, probing
			// keeps the retry loop invisible while the owning model is still loading (503)
			const status = await ChatService.probeResumeStatus(streamId);

			if (status === 503) {
				// make the wait visible: the empty assistant row persisted at send time renders
				// the processing info, whose model load percentage flows from the models feed
				this.resumePendingConvs.add(convId);
				this.host.setChatLoading(convId, true);

				if (!this.resumeRetryTimers.has(convId)) {
					this.resumeRetryTimers.set(
						convId,
						setTimeout(() => {
							this.resumeRetryTimers.delete(convId);
							void this.discoverActiveStream(convId);
						}, STREAM_RESUME_RETRY_MS)
					);
				}

				return;
			}

			if (this.resumePendingConvs.delete(convId) && status !== 200) {
				// the wait is over without a session to attach, drop the visible loading state
				this.host.setChatLoading(convId, false);
			}

			if (status === 0) {
				// transient network failure, the next mount or visibility change retries
				return;
			}

			if (status !== 200) {
				// the session is gone (stopped, TTL expired), nothing to resume anymore
				ChatService.clearStreamState(convId);

				return;
			}

			await this.attachServerStream(convId, streamId);

			// if attachServerStream failed (session gone, TTL expired), clear the local state to avoid retrying forever
			if (!this.host.chatStreamingStates.has(convId) && !this.host.activity.isLocal(convId)) {
				ChatService.clearStreamState(convId);
			}
		} finally {
			this.discoveringConvs.delete(convId);
		}
	}

	/**
	 * Model frozen at send time for a stream awaiting resume, from the persisted stream state.
	 * The load progress indicator targets it after a reload, when the message row has no model
	 * yet and the dropdown selection may not be restored.
	 */
	getResumeModel(convId: string): string | null {
		return ChatService.getStreamState(convId)?.model ?? null;
	}

	/**
	 * Resync the activity ledger's remote set from the backend. Called by the layout at mount and
	 * on visibilitychange, no polling. A snapshot semantic: stale entries for sessions that
	 * finalized while the browser was elsewhere are dropped naturally.
	 */
	async syncRemoteRunningStreams(): Promise<void> {
		// the conversations store loads from IndexedDB asynchronously, the +layout onMount caller
		// fires before that finishes. read ids straight from the DB so the result does not depend
		// on the store init race, and the sidebar spinners light up at first paint for every conv
		// the user owns even if it has not been hydrated into the store yet
		let ids: string[];

		try {
			const all = await DatabaseService.getAllConversations();

			ids = all.map((c) => c.id).filter((id) => !!id);
		} catch (e) {
			console.warn('syncRemoteRunningStreams DB read failed:', e);

			return;
		}

		// only ask about conv ids the user already owns
		if (ids.length === 0) {
			this.host.activity.applyRemoteSnapshot([]);

			return;
		}

		// rebuild the frozen conv::model identity per conv so a session started with a model still
		// matches. the server response is mapped back to the bare id below for the sidebar set
		const lookupIds = ids.map((id) =>
			ChatService.resumeStreamIdentity(id, ChatService.getStreamState(id), null)
		);

		let sessions: ApiStreamSession[];

		try {
			sessions = await ChatService.lookupStreamSessions(lookupIds);
		} catch (e) {
			console.warn('syncRemoteRunningStreams lookup failed:', e);

			return;
		}
		const running = new SvelteSet<string>();

		for (const s of sessions) {
			if (s && !s.is_done && typeof s.conversation_id === 'string' && s.conversation_id) {
				// strip the optional ::model suffix, the sidebar set is keyed by the bare conv id
				const sepIdx = s.conversation_id.indexOf(CONVERSATION_ID_SEPARATOR);
				const bareId = sepIdx === -1 ? s.conversation_id : s.conversation_id.slice(0, sepIdx);

				running.add(bareId);
			}
		}
		this.host.activity.applyRemoteSnapshot(running);
	}

	private async attachServerStream(convId: string, streamId?: string): Promise<void> {
		if (!convId) return;

		if (this.host.chatStreamingStates.has(convId)) return;

		// flip the spinner immediately, the user sees activity as soon as the conv becomes active
		this.host.setChatLoading(convId, true);

		// only set the active processing conv if we are looking at it, otherwise a background
		// attach would steal the indicator from the conv the user is currently viewing
		if (convId === conversationsStore.activeConversation?.id) {
			this.host.processing.setActiveConversation(convId);
		}

		const unlock = () => {
			this.host.setChatLoading(convId, false);
			this.host.clearChatStreaming(convId);
		};
		// fetch the replay stream from byte 0, rebuild the assistant message from scratch.
		// resolve the server side identity, fall back to streamIdentity when the caller does not
		// pass a streamId. probeServerStream returns the full id (with ::model suffix when present)
		const id = streamId || streamIdentity(convId, modelsStore.selectedModelName);

		let response: Response;

		try {
			response = await ChatService.fetchStreamReplay(id);
		} catch (e) {
			console.error(`attachServerStream replay failed for conv ${convId}:`, e);
			unlock();

			return;
		}

		// load the target conversation messages by id, not via the active store. when multiple
		// attaches run in parallel the active store may reflect another conv and writing through
		// its index mixes content across convs (CoT flicker, message bleed). by going through the
		// DB we stay isolated, and only mirror into the active store when the attached conv is
		// the one currently displayed
		let messages: DatabaseMessage[];

		try {
			messages = await DatabaseService.getConversationMessages(convId);
		} catch (e) {
			console.error('attachServerStream load messages failed:', e);
			unlock();

			return;
		}

		// locate the slot to splice into, create a placeholder assistant message if there is none.
		// we use the conv-scoped findLastAssistantIdx helpers, they only depend on the array
		let targetIdx = this.findLastAssistantIdx(messages);

		if (targetIdx === -1) {
			const lastUserIdx = this.findLastUserIdx(messages);

			if (lastUserIdx === -1) {
				console.warn(
					`attachServerStream: conv ${convId} has no user or assistant message, cannot splice`
				);
				unlock();

				return;
			}

			try {
				const placeholder = await DatabaseService.createMessageBranch(
					{
						children: [],
						content: '',
						convId,
						parent: messages[lastUserIdx].id,
						role: MessageRole.ASSISTANT,
						timestamp: Date.now(),
						toolCalls: '',
						type: MessageType.TEXT
					} as Omit<DatabaseMessage, 'id'>,
					messages[lastUserIdx].id
				);

				messages = [...messages, placeholder];
				targetIdx = messages.length - 1;

				// only push into the active store when this conv is the one displayed right now
				if (convId === conversationsStore.activeConversation?.id) {
					conversationsStore.addMessageToActive(placeholder);
				}
			} catch (e) {
				console.error('attachServerStream placeholder creation failed:', e);
				unlock();

				return;
			}
		}

		if (targetIdx === -1) {
			unlock();

			return;
		}

		const targetMessage = messages[targetIdx];
		const targetMessageId = targetMessage.id;
		// when the assistant slot already has content, the running session is a continue or
		// another append flow and its buffer holds only the appended deltas. preserve the prefix
		// and let the replay add to it. when the slot is empty the session buffer holds the whole
		// message so we wipe and rebuild from byte 0
		const existingContent = targetMessage.content ?? '';
		const existingReasoning = targetMessage.reasoningContent ?? '';
		const isAppendMode = existingContent.length > 0;
		// helper: write to the active store only when the attached conv is currently displayed.
		// the lookup by message id is robust to reordering of activeMessages, two parallel attaches
		// can no longer step on each other's indices
		const writeActive = (updates: Partial<DatabaseMessage>) => {
			if (convId !== conversationsStore.activeConversation?.id) {
				return;
			}

			const liveIdx = conversationsStore.findMessageIndex(targetMessageId);

			if (liveIdx === -1) return;

			conversationsStore.updateMessageAtIndex(liveIdx, updates);
		};

		if (!isAppendMode) {
			writeActive({ content: '', reasoningContent: undefined });
		}

		// extract the model suffix, the resume calls in handleStreamResponse must reuse the model
		// the session was tagged with, not the live dropdown
		const sepIdx = id.indexOf(CONVERSATION_ID_SEPARATOR);
		const attachedModel: string | null = sepIdx === -1 ? null : id.slice(sepIdx + 2);

		this.host.setChatStreaming(convId, existingContent, targetMessageId, attachedModel);
		const abortController = this.host.getOrCreateAbortController(convId);

		let streamedContent = '';
		let streamedReasoningContent = '';

		const cleanup = () => {
			unlock();
			this.host.processing.setState(convId, null);
		};

		try {
			await ChatService.handleStreamResponse(
				response,
				(chunk: string) => {
					streamedContent += chunk;
					const displayed = isAppendMode ? existingContent + streamedContent : streamedContent;

					writeActive({ content: displayed });
					this.host.setChatStreaming(convId, displayed, targetMessageId);
				},
				async (
					finalContent?: string,
					reasoningContent?: string,
					timings?: ChatMessageTimings,
					toolCalls?: string
				) => {
					const streamed = streamedContent || finalContent || '';
					const streamedR = streamedReasoningContent || reasoningContent || '';
					const content = isAppendMode ? existingContent + streamed : streamed;
					const reasoning = isAppendMode ? existingReasoning + streamedR : streamedR;

					// the DB write is the source of truth, mirror to the active store only when
					// the conv is currently displayed
					await DatabaseService.updateMessage(targetMessageId, {
						content,
						reasoningContent: reasoning || undefined,
						timings,
						toolCalls: toolCalls || ''
					});
					writeActive({
						content,
						reasoningContent: reasoning || undefined,
						timings
					});
					cleanup();
				},
				(err: Error) => {
					console.error('attachServerStream pipe error:', err);
					cleanup();
				},
				(chunk: string) => {
					streamedReasoningContent += chunk;
					const displayed = isAppendMode
						? existingReasoning + streamedReasoningContent
						: streamedReasoningContent;

					writeActive({ reasoningContent: displayed });
				},
				undefined,
				undefined,
				undefined,
				undefined,
				convId,
				abortController.signal,
				(connState: StreamConnectionState) => {
					if (convId === conversationsStore.activeConversation?.id) {
						this.host.streamConnectionState = connState;
					}
				},
				attachedModel
			);
		} catch (e) {
			console.error('attachServerStream pipe crashed:', e);
			cleanup();
		}
	}

	private findLastAssistantIdx(messages: DatabaseMessage[]): number {
		for (let i = messages.length - 1; i >= 0; i--) {
			if (messages[i].role === MessageRole.ASSISTANT) return i;
		}

		return -1;
	}

	private findLastUserIdx(messages: DatabaseMessage[]): number {
		for (let i = messages.length - 1; i >= 0; i--) {
			if (messages[i].role === MessageRole.USER) return i;
		}

		return -1;
	}

	/**
	 * Server side stream discovery, split in three pieces:
	 *
	 * probeServerStream(convId) -> hits POST /v1/streams/lookup with the conv id, returns the session to attach
	 *   to or null. Pure read, no side effect, no UI lock. Safe to fire in parallel with anything.
	 *
	 * attachServerStream(convId) -> flips the spinner immediately, fetches the replay stream
	 *   from byte 0, finds the assistant slot to splice into (creates a placeholder if the conv has
	 *   no assistant message yet, for cross device or fresh local DB cases), and pipes the SSE bytes
	 *   into the message via handleStreamResponse.
	 *
	 * discoverActiveStream(convId) -> probe + attach in one call. Used by callers that do not need
	 *   to overlap the probe with other async work.
	 *
	 * The chat page in +page.svelte calls discoverActiveStream once the conversation is active
	 * (immediately if it already is, after loadConversation settles otherwise), and re-runs it on
	 * visibilitychange. Attaching only after the conversation is loaded gives the earliest
	 * possible time to spinner and avoids racing against an empty activeMessages array.
	 */
	private async probeServerStream(convId: string): Promise<ApiStreamSession | null> {
		if (!convId) return null;

		let sessions: ApiStreamSession[];

		try {
			sessions = await ChatService.lookupStreamSessions([convId]);
		} catch (e) {
			console.warn(`probeServerStream failed for conv ${convId}:`, e);

			return null;
		}

		return ChatService.selectActiveStream(sessions);
	}
}
