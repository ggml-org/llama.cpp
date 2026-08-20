/**
 * AgenticStore - Multi-turn agentic loop orchestration
 *
 * Drives the agentic loop over MCP tools: streams each LLM turn, detects
 * tool calls, executes them via mcpStore, and enforces the turn limit. Each
 * turn produces one assistant message (with tool_calls) and one tool result
 * message per executed call, persisted as separate DB rows.
 *
 * Uses ChatService for streaming and mcpStore for tool execution; waits on
 * the permission/continue/steering gates owned by {@link AgenticGates}.
 */

import { DEFAULT_AGENTIC_CONFIG, NEWLINE } from '$lib/constants';
import {
	AUDIO_MIME_TO_EXTENSION,
	DATA_URI_BASE64_REGEX,
	DEFAULT_AUDIO_EXTENSION,
	DEFAULT_IMAGE_EXTENSION,
	IMAGE_MIME_TO_EXTENSION,
	MCP_ATTACHMENT_NAME_PREFIX,
	MIME_TYPE_PREFIXES
} from '$lib/constants';
import { BuiltInTool, ToolPermissionDecision, ToolSource } from '$lib/enums';
import {
	AttachmentType,
	ContentPartType,
	MessageRole,
	MimeTypePrefix,
	ToolCallType
} from '$lib/enums';
import { ChatService } from '$lib/services';
import { ReadMediaService } from '$lib/services/read-media.service';
import { SandboxService } from '$lib/services/sandbox.service';
import { ToolsService } from '$lib/services/tools.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { AgenticGates } from '$lib/stores/agentic/gates.svelte';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import { mcpStore } from '$lib/stores/mcp/index.svelte';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import type {
	AgenticConfig,
	AgenticFlowParams,
	AgenticFlowResult,
	AgenticSession,
	McpServerOverride,
	MCPToolCall,
	SettingsConfigType,
	ToolExecutionResult
} from '$lib/types';
import type {
	AgenticFlowCallbacks,
	AgenticFlowOptions,
	AgenticMessage,
	AgenticToolCallList,
	SteeringMessage
} from '$lib/types/agentic';
import type {
	ApiChatCompletionToolCall,
	ApiChatMessageContentPart,
	ApiChatMessageData
} from '$lib/types/api';
import type {
	ChatMessageAgenticTimings,
	ChatMessageAgenticTurnStats,
	ChatMessagePromptProgress,
	ChatMessageTimings,
	ChatMessageToolCallTiming
} from '$lib/types/chat';
import type {
	DatabaseMessage,
	DatabaseMessageExtra,
	DatabaseMessageExtraAudioFile,
	DatabaseMessageExtraImageFile
} from '$lib/types/database';
import {
	executeBrowserInfoTool,
	executeGetDatetimeTool,
	getAudioInputFormat,
	isAbortError
} from '$lib/utils';
import { SvelteMap } from 'svelte/reactivity';

function createDefaultSession(): AgenticSession {
	return {
		currentTurn: 0,
		executingToolCallId: null,
		flowRootMessageId: null,
		isRunning: false,
		lastError: null,
		liveLlm: null,
		pendingPermissionRequest: null,
		streamingToolCall: null,
		totalToolCalls: 0
	};
}

function toAgenticMessages(messages: ApiChatMessageData[]): AgenticMessage[] {
	return messages.map((message) => {
		if (
			message.role === MessageRole.ASSISTANT &&
			message.tool_calls &&
			message.tool_calls.length > 0
		) {
			return {
				content: message.content,
				reasoning_content: message.reasoning_content,
				role: MessageRole.ASSISTANT,
				tool_calls: message.tool_calls.map((call, index) => ({
					function: {
						arguments: call.function?.arguments ?? '',
						name: call.function?.name ?? ''
					},
					id: call.id ?? `call_${index}`,
					type: (call.type as ToolCallType.FUNCTION) ?? ToolCallType.FUNCTION
				}))
			} satisfies AgenticMessage;
		}

		if (message.role === MessageRole.ASSISTANT) {
			return {
				content: message.content,
				reasoning_content: message.reasoning_content,
				role: MessageRole.ASSISTANT
			} satisfies AgenticMessage;
		}

		if (message.role === MessageRole.TOOL && message.tool_call_id) {
			return {
				content: typeof message.content === 'string' ? message.content : '',
				role: MessageRole.TOOL,
				tool_call_id: message.tool_call_id
			} satisfies AgenticMessage;
		}

		return {
			content: message.content,
			role: message.role as MessageRole.SYSTEM | MessageRole.USER
		} satisfies AgenticMessage;
	});
}

class AgenticStore {
	// permission, continue and steering gates the loop waits on between turns
	private gates = new AgenticGates();
	private sessions = new SvelteMap<string, AgenticSession>();

	get isAnyRunning(): boolean {
		for (const session of this.sessions.values()) {
			if (session.isRunning) return true;
		}

		return false;
	}

	get isReady(): boolean {
		return true;
	}

	clearError(conversationId: string): void {
		this.updateSession(conversationId, { lastError: null });
	}

	clearSession(conversationId: string): void {
		this.sessions.delete(conversationId);
	}

	/**
	 * Clear the pending steering message without consuming it.
	 */
	clearSteeringMessage(conversationId: string): void {
		this.gates.clearSteeringMessage(conversationId);
	}

	constructor() {
		// drop per-conversation session state when the conversation is deleted,
		// otherwise every conversation that ever ran a flow leaks a session here
		conversationsStore.onConversationsDeleted((convIds) => {
			for (const convId of convIds) {
				this.sessions.delete(convId);
			}
		});
	}

	/**
	 * Consume and return the pending steering message for re-sending.
	 * Called by chatStore after the agentic flow exits.
	 */
	consumePendingSteeringMessage(conversationId: string): SteeringMessage | null {
		return this.gates.consumePendingSteeringMessage(conversationId);
	}

	getActiveSessions(): Array<{ conversationId: string; session: AgenticSession }> {
		const active: Array<{ conversationId: string; session: AgenticSession }> = [];

		for (const [conversationId, session] of this.sessions.entries()) {
			if (session.isRunning) active.push({ conversationId, session });
		}

		return active;
	}

	getConfig(settings: SettingsConfigType, perChatOverrides?: McpServerOverride[]): AgenticConfig {
		const maxTurns = Number(settings.agenticMaxTurns) || DEFAULT_AGENTIC_CONFIG.maxTurns;
		const hasTools =
			mcpStore.hasEnabledServers(perChatOverrides) ||
			toolsStore.serverTools.length > 0 ||
			toolsStore.browserTools.length > 0 ||
			toolsStore.customTools.length > 0;

		return {
			enabled: hasTools && DEFAULT_AGENTIC_CONFIG.enabled,
			maxTurns
		};
	}

	getCurrentTurn(conversationId: string): number {
		return this.sessions.get(conversationId)?.currentTurn ?? 0;
	}

	getExecutingToolCallId(conversationId: string): string | null {
		return this.sessions.get(conversationId)?.executingToolCallId ?? null;
	}

	// read-only: safe to call from derivations, unlike getSession
	getFlowRootMessageId(conversationId: string): string | null {
		return this.sessions.get(conversationId)?.flowRootMessageId ?? null;
	}

	getLastError(conversationId: string): Error | null {
		return this.sessions.get(conversationId)?.lastError ?? null;
	}

	// read-only: safe to call from derivations, unlike getSession
	getLiveLlmTotals(conversationId: string): AgenticSession['liveLlm'] {
		return this.sessions.get(conversationId)?.liveLlm ?? null;
	}

	getPendingContinueRequest(conversationId: string): boolean {
		return this.gates.getPendingContinueRequest(conversationId);
	}

	getPendingPermissionRequest(
		conversationId: string
	): { toolName: string; serverLabel: string } | null {
		return this.gates.getPendingPermissionRequest(conversationId);
	}

	getPendingSteeringMessageContent(conversationId: string): string | null {
		return this.gates.getPendingSteeringMessageContent(conversationId);
	}

	getPendingSteeringMessageExtras(conversationId: string): DatabaseMessageExtra[] | undefined {
		return this.gates.getPendingSteeringMessageExtras(conversationId);
	}

	getSession(conversationId: string): AgenticSession {
		let session = this.sessions.get(conversationId);

		if (!session) {
			session = createDefaultSession();
			this.sessions.set(conversationId, session);
		}

		return session;
	}

	getStreamingToolCall(conversationId: string): { name: string; arguments: string } | null {
		return this.sessions.get(conversationId)?.streamingToolCall ?? null;
	}

	getTotalToolCalls(conversationId: string): number {
		return this.sessions.get(conversationId)?.totalToolCalls ?? 0;
	}

	hasPendingSteeringMessage(conversationId: string): boolean {
		return this.gates.hasPendingSteeringMessage(conversationId);
	}

	/**
	 * Queue a steering message. When the current agentic turn completes,
	 * the flow exits and the caller re-sends the message as a normal chat message.
	 */
	injectSteeringMessage(
		conversationId: string,
		content: string,
		extras?: DatabaseMessageExtra[]
	): void {
		this.gates.injectSteeringMessage(conversationId, content, extras);
	}

	isRunning(conversationId: string): boolean {
		return this.sessions.get(conversationId)?.isRunning ?? false;
	}

	resolveContinue(conversationId: string, shouldContinue: boolean): void {
		this.gates.resolveContinue(conversationId, shouldContinue);
	}

	resolvePermission(conversationId: string, decision: ToolPermissionDecision): void {
		this.gates.resolvePermission(conversationId, decision);
	}

	async runAgenticFlow(params: AgenticFlowParams): Promise<AgenticFlowResult> {
		const {
			callbacks,
			conversationId,
			flowRootMessageId,
			messages,
			options = {},
			perChatOverrides,
			signal
		} = params;

		// Clear any pending permissions/continue requests for this conversation when starting a new flow
		this.gates.clear(conversationId);

		// Ensure server tools are fetched before checking if agentic is enabled
		if (toolsStore.serverTools.length === 0 && !toolsStore.loading) {
			await toolsStore.fetchServerTools();
		}

		const agenticConfig = this.getConfig(settingsStore.config, perChatOverrides);

		if (!agenticConfig.enabled) return { handled: false };

		const hasMcpServers = mcpStore.hasEnabledServers(perChatOverrides);

		if (hasMcpServers) {
			const initialized = await mcpStore.ensureInitialized(perChatOverrides);

			if (!initialized) {
				console.log('[AgenticStore] MCP not initialized');
			}
		}

		const tools = toolsStore.getEnabledToolsForLLM();

		if (tools.length === 0) {
			return { handled: false };
		}

		console.log(`[AgenticStore] Starting agentic flow with ${tools.length} tools`);

		const normalizedMessages: ApiChatMessageData[] =
			await ChatService.normalizeMessagesForApi(messages);

		this.updateSession(conversationId, {
			currentTurn: 0,
			flowRootMessageId: flowRootMessageId ?? null,
			isRunning: true,
			lastError: null,
			liveLlm: null,
			totalToolCalls: 0
		});

		if (hasMcpServers) mcpStore.acquireConnection();

		try {
			await this.executeAgenticLoop({
				agenticConfig,
				callbacks,
				conversationId,
				messages: normalizedMessages,
				options,
				signal,
				tools
			});

			return { handled: true };
		} catch (error) {
			const normalizedError = error instanceof Error ? error : new Error(String(error));

			this.updateSession(conversationId, { lastError: normalizedError });
			callbacks.onError?.(normalizedError);

			return { error: normalizedError, handled: true };
		} finally {
			this.updateSession(conversationId, {
				flowRootMessageId: null,
				isRunning: false,
				liveLlm: null
			});

			if (hasMcpServers) {
				await mcpStore
					.releaseConnection()
					.catch((err: unknown) =>
						console.warn('[AgenticStore] Failed to release MCP connection:', err)
					);
			}
		}
	}

	private buildAttachmentName(mimeType: string, index: number): string {
		const extension = mimeType.startsWith(MimeTypePrefix.AUDIO)
			? (AUDIO_MIME_TO_EXTENSION[mimeType] ?? DEFAULT_AUDIO_EXTENSION)
			: (IMAGE_MIME_TO_EXTENSION[mimeType] ?? DEFAULT_IMAGE_EXTENSION);

		return `${MCP_ATTACHMENT_NAME_PREFIX}-${Date.now()}-${index}.${extension}`;
	}

	private buildFinalTimings(
		capturedTimings: ChatMessageTimings | undefined,
		agenticTimings: ChatMessageAgenticTimings
	): ChatMessageTimings | undefined {
		if (agenticTimings.toolCallsCount === 0) return capturedTimings;

		return {
			agentic: agenticTimings,
			cache_n: capturedTimings?.cache_n,
			predicted_ms: capturedTimings?.predicted_ms,
			predicted_n: capturedTimings?.predicted_n,
			prompt_ms: capturedTimings?.prompt_ms,
			prompt_n: capturedTimings?.prompt_n
		};
	}

	private async executeAgenticLoop(params: {
		conversationId: string;
		messages: ApiChatMessageData[];
		options: AgenticFlowOptions;
		tools: ReturnType<typeof toolsStore.getEnabledToolsForLLM>;
		agenticConfig: AgenticConfig;
		callbacks: AgenticFlowCallbacks;
		signal?: AbortSignal;
	}): Promise<void> {
		const { agenticConfig, callbacks, conversationId, messages, options, signal, tools } = params;
		const {
			createAssistantMessage,
			createToolResultMessage,
			onAssistantTurnComplete,
			onAttachments,
			onChunk,
			onCompletionId,
			onFlowComplete,
			onModel,
			onReasoningChunk,
			onTimings,
			onToolCallsStreaming,
			onTurnComplete,
			updateToolResultMessage
		} = callbacks;
		const sessionMessages: AgenticMessage[] = toAgenticMessages(messages);

		let capturedTimings: ChatMessageTimings | undefined;
		let totalToolCallCount = 0;

		const agenticTimings: ChatMessageAgenticTimings = {
			llm: { predicted_ms: 0, predicted_n: 0, prompt_ms: 0, prompt_n: 0 },
			perTurn: [],
			toolCalls: [],
			toolCallsCount: 0,
			toolsMs: 0,
			turns: 0
		};
		const maxTurns = agenticConfig.maxTurns;
		const effectiveModel = options.model || modelsStore.models[0]?.model || '';

		let turn = 0;

		while (true) {
			if (turn >= maxTurns) {
				// Turn limit reached - ask user whether to continue
				const shouldContinue = await this.gates.requestContinue(conversationId, signal);

				// Yield to allow Svelte to flush the UI update
				await new Promise((r) => setTimeout(r, 0));

				if (!shouldContinue || signal?.aborted) {
					onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

					return;
				}

				// User chose to continue - extend the limit
				turn = 0;
			}

			this.updateSession(conversationId, { currentTurn: turn + 1 });
			agenticTimings.turns = turn + 1;

			if (signal?.aborted) {
				onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

				return;
			}

			// For turns > 0, create a new assistant message via callback
			if (turn > 0 && createAssistantMessage) {
				await createAssistantMessage();
			}

			let turnContent = '';
			let turnReasoningContent = '';
			let turnToolCalls: ApiChatCompletionToolCall[] = [];
			let lastStreamingToolCallName = '';
			let lastStreamingToolCallArgsLength = 0;
			let turnTimings: ChatMessageTimings | undefined;

			const turnStats: ChatMessageAgenticTurnStats = {
				llm: { predicted_ms: 0, predicted_n: 0, prompt_ms: 0, prompt_n: 0 },
				toolCalls: [],
				toolsMs: 0,
				turn: turn + 1
			};

			try {
				await ChatService.sendMessage(
					sessionMessages as ApiChatMessageData[],
					{
						...options,
						onChunk: (chunk: string) => {
							turnContent += chunk;
							onChunk?.(chunk);
						},
						onComplete: () => {
							/* Completion handled after sendMessage resolves */
						},
						onCompletionId,
						onError: (error: Error) => {
							throw error;
						},
						onModel,
						onReasoningChunk: (chunk: string) => {
							turnReasoningContent += chunk;
							onReasoningChunk?.(chunk);
						},
						onTimings: (timings?: ChatMessageTimings, progress?: ChatMessagePromptProgress) => {
							onTimings?.(timings, progress);

							if (timings) {
								capturedTimings = timings;
								turnTimings = timings;

								// completed turns + in-flight turn live counts
								this.updateSession(conversationId, {
									liveLlm: {
										predicted_ms: agenticTimings.llm.predicted_ms + (timings.predicted_ms ?? 0),
										predicted_n: agenticTimings.llm.predicted_n + (timings.predicted_n ?? 0),
										prompt_ms: agenticTimings.llm.prompt_ms + (timings.prompt_ms ?? 0),
										prompt_n: agenticTimings.llm.prompt_n + (timings.prompt_n ?? 0)
									}
								});
							}
						},
						onToolCallChunk: (serialized: string) => {
							try {
								turnToolCalls = JSON.parse(serialized) as ApiChatCompletionToolCall[];

								onToolCallsStreaming?.(turnToolCalls);

								if (turnToolCalls.length > 0 && turnToolCalls[0]?.function) {
									const name = turnToolCalls[0].function.name || '';
									const args = turnToolCalls[0].function.arguments || '';
									const argsLengthBucket = Math.floor(args.length / 100);

									if (
										name !== lastStreamingToolCallName ||
										argsLengthBucket !== lastStreamingToolCallArgsLength
									) {
										lastStreamingToolCallName = name;
										lastStreamingToolCallArgsLength = argsLengthBucket;
										this.updateSession(conversationId, {
											streamingToolCall: { arguments: args, name }
										});
									}
								}
							} catch {
								/* Ignore parse errors during streaming */
							}
						},
						stream: true,
						tools: tools.length > 0 ? tools : undefined
					},
					conversationId,
					signal
				);

				this.updateSession(conversationId, { streamingToolCall: null });

				if (turnTimings) {
					agenticTimings.llm.predicted_n += turnTimings.predicted_n || 0;
					agenticTimings.llm.predicted_ms += turnTimings.predicted_ms || 0;
					agenticTimings.llm.prompt_n += turnTimings.prompt_n || 0;
					agenticTimings.llm.prompt_ms += turnTimings.prompt_ms || 0;
					turnStats.llm.predicted_n = turnTimings.predicted_n || 0;
					turnStats.llm.predicted_ms = turnTimings.predicted_ms || 0;
					turnStats.llm.prompt_n = turnTimings.prompt_n || 0;
					turnStats.llm.prompt_ms = turnTimings.prompt_ms || 0;
				}
			} catch (error) {
				if (signal?.aborted) {
					// Save whatever we have for this turn before exiting
					await onAssistantTurnComplete?.(
						turnContent,
						turnReasoningContent || undefined,
						this.buildFinalTimings(capturedTimings, agenticTimings),
						undefined
					);
					onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

					return;
				}

				const normalizedError = error instanceof Error ? error : new Error('LLM stream error');

				// preserve partial output as is, the outer error dialog informs the user separately
				await onAssistantTurnComplete?.(
					turnContent,
					turnReasoningContent || undefined,
					this.buildFinalTimings(capturedTimings, agenticTimings),
					undefined
				);
				onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

				throw normalizedError;
			}

			// If the abort landed while ChatService.sendMessage was still resolving, the
			// outer catch above never fires because ChatService swallows the AbortError
			// and returns normally. Bail out here so a half-received tool_call (truncated
			// arguments JSON) is not persisted as if it were complete.
			if (signal?.aborted) {
				await onAssistantTurnComplete?.(
					turnContent,
					turnReasoningContent || undefined,
					this.buildFinalTimings(capturedTimings, agenticTimings),
					undefined
				);
				onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

				return;
			}

			// === Steering check: if a user message was queued during this turn, exit the flow.
			// The caller (chatStore) will consume the pending message and re-send it normally.
			if (this.gates.hasPendingSteeringMessage(conversationId)) {
				console.log('[AgenticStore] Steering message detected after turn, exiting agentic flow');
				await onAssistantTurnComplete?.(
					turnContent,
					turnReasoningContent || undefined,
					this.buildFinalTimings(capturedTimings, agenticTimings),
					turnToolCalls.length > 0 ? this.normalizeToolCalls(turnToolCalls) : undefined
				);
				onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

				return;
			}

			// No tool calls = final turn, save and complete
			if (turnToolCalls.length === 0) {
				agenticTimings.perTurn!.push(turnStats);

				const finalTimings = this.buildFinalTimings(capturedTimings, agenticTimings);

				await onAssistantTurnComplete?.(
					turnContent,
					turnReasoningContent || undefined,
					finalTimings,
					undefined
				);

				if (finalTimings) onTurnComplete?.(finalTimings);

				onFlowComplete?.(finalTimings);

				return;
			}

			// Normalize and save assistant turn with tool calls
			const normalizedCalls = this.normalizeToolCalls(turnToolCalls);

			if (normalizedCalls.length === 0) {
				await onAssistantTurnComplete?.(
					turnContent,
					turnReasoningContent || undefined,
					this.buildFinalTimings(capturedTimings, agenticTimings),
					undefined
				);
				onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

				return;
			}

			totalToolCallCount += normalizedCalls.length;
			this.updateSession(conversationId, { totalToolCalls: totalToolCallCount });

			// Save the assistant message with its tool calls
			await onAssistantTurnComplete?.(
				turnContent,
				turnReasoningContent || undefined,
				turnTimings,
				normalizedCalls
			);

			// Add assistant message to session history
			sessionMessages.push({
				content: turnContent || undefined,
				reasoning_content: turnReasoningContent || undefined,
				role: MessageRole.ASSISTANT,
				tool_calls: normalizedCalls
			});

			// Execute each tool call and create result messages
			for (let i = 0; i < normalizedCalls.length; i++) {
				const toolCall = normalizedCalls[i];

				if (signal?.aborted) {
					onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

					return;
				}

				// Check for pending steering message - skip remaining tool calls
				if (this.gates.hasPendingSteeringMessage(conversationId)) {
					console.log(
						`[AgenticStore] Steering message detected, skipping ${normalizedCalls.length - i} remaining tool call(s)`
					);
					for (let j = i; j < normalizedCalls.length; j++) {
						const remainingCall = normalizedCalls[j];
						const interruptedContent = 'Tool execution was interrupted by a new user message.';

						if (createToolResultMessage) {
							await createToolResultMessage(remainingCall.id, interruptedContent);
						}

						sessionMessages.push({
							content: interruptedContent,
							role: MessageRole.TOOL,
							tool_call_id: remainingCall.id
						});
					}

					break;
				}

				const toolName = toolCall.function.name;
				const serverLabel = toolsStore.getToolServerLabel(toolName);
				// Ask for permission before executing the tool
				const permission = await this.gates.requestPermission(
					conversationId,
					toolName,
					serverLabel,
					signal
				);

				// Yield to allow Svelte to flush the UI update (hide permission dialog)
				await new Promise((r) => setTimeout(r, 0));

				if (signal?.aborted) {
					onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

					return;
				}

				const toolStartTime = performance.now();
				const toolSource = toolsStore.getToolSource(toolName);

				let result = '';
				let toolSuccess = true;
				let createdToolResultMessageId: string | null = null;

				// Streaming tools (currently only exec_shell_command): mark
				// the session so the matching renderer can switch to live mode.
				// Cleared unconditionally below.
				this.updateSession(conversationId, { executingToolCallId: toolCall.id });

				if (permission === ToolPermissionDecision.DENY) {
					result = 'Tool execution was denied by the user.';
					toolSuccess = false;
				} else {
					try {
						if (
							toolSource === ToolSource.SERVER &&
							toolName === BuiltInTool.SERVER_EXEC_SHELL_COMMAND &&
							createToolResultMessage &&
							updateToolResultMessage
						) {
							const args = this.parseToolArguments(toolCall.function.arguments);
							const cwd = conversationsStore.activeConversation?.cwd;
							const msg = await createToolResultMessage(toolCall.id, '', undefined, cwd);

							createdToolResultMessageId = msg.id;

							let accumulated = '';

							for await (const ev of ToolsService.streamTool(toolName, args, signal, cwd)) {
								if (ev.chunk !== null) {
									accumulated += ev.chunk;
									await updateToolResultMessage(msg.id, accumulated);
								}

								if (ev.done) {
									if (ev.error) {
										accumulated = accumulated
											? `${accumulated}\nError: ${ev.error}`
											: `Error: ${ev.error}`;
										await updateToolResultMessage(msg.id, accumulated);
										toolSuccess = false;
									}

									break;
								}
							}
							result = accumulated;
						} else if (toolSource === ToolSource.SERVER) {
							const args = this.parseToolArguments(toolCall.function.arguments);
							const cwd = conversationsStore.activeConversation?.cwd;
							const executionResult = await ToolsService.executeTool(toolName, args, signal, cwd);

							result = executionResult.content;

							if (executionResult.isError) toolSuccess = false;
						} else if (toolSource === ToolSource.BROWSER) {
							const args = this.parseToolArguments(toolCall.function.arguments);

							let executionResult: ToolExecutionResult;

							if (toolName === BuiltInTool.BROWSER_GET_DATETIME) {
								executionResult = executeGetDatetimeTool();
							} else if (toolName === BuiltInTool.SERVER_GET_INFO) {
								executionResult = executeBrowserInfoTool();
							} else if (toolName === BuiltInTool.BROWSER_READ_MEDIA) {
								executionResult = await ReadMediaService.executeTool(
									args,
									{
										audio: modelsStore.props.modelSupportsAudio(effectiveModel),
										vision: modelsStore.props.modelSupportsVision(effectiveModel)
									},
									signal,
									conversationsStore.activeConversation?.cwd
								);
							} else {
								executionResult = await SandboxService.executeTool(toolName, args, signal);
							}

							result = executionResult.content;

							if (executionResult.isError) toolSuccess = false;
						} else {
							const mcpCall: MCPToolCall = {
								function: { arguments: toolCall.function.arguments, name: toolName },
								id: toolCall.id
							};
							const executionResult = await mcpStore.executeTool(mcpCall, signal);

							result = executionResult.content;
						}
					} catch (error) {
						if (isAbortError(error)) {
							this.updateSession(conversationId, { executingToolCallId: null });
							onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

							return;
						}

						// Carry the partial stream contents already mirrored to the UI -
						// they show up as live output even if the stream broke off mid-run.
						result = result
							? `${result}\nError: ${error instanceof Error ? error.message : String(error)}`
							: `Error: ${error instanceof Error ? error.message : String(error)}`;
						toolSuccess = false;

						if (createdToolResultMessageId && updateToolResultMessage) {
							await updateToolResultMessage(createdToolResultMessageId, result);
						}
					}
				}

				this.updateSession(conversationId, { executingToolCallId: null });

				const toolDurationMs = performance.now() - toolStartTime;
				const toolTiming: ChatMessageToolCallTiming = {
					duration_ms: Math.round(toolDurationMs),
					name: toolCall.function.name,
					success: toolSuccess
				};

				agenticTimings.toolCalls!.push(toolTiming);
				agenticTimings.toolCallsCount++;
				agenticTimings.toolsMs += Math.round(toolDurationMs);
				turnStats.toolCalls.push(toolTiming);
				turnStats.toolsMs += Math.round(toolDurationMs);

				if (signal?.aborted) {
					onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

					return;
				}

				const { attachments, cleanedResult } = this.extractBase64Attachments(result);

				// For streaming tools the result message was created empty
				// at the start of execution and updated in place as chunks
				// arrived via updateToolResultMessage. Skip the second
				// create call - just attach any base64 attachments found in
				// the final accumulator (rare, since chunks usually don't
				// carry image data URIs) and emit the attachments callback.
				let toolResultMessage: DatabaseMessage | undefined;

				if (createdToolResultMessageId) {
					toolResultMessage = { id: createdToolResultMessageId } as DatabaseMessage;

					if (attachments.length > 0 && updateToolResultMessage) {
						await updateToolResultMessage(createdToolResultMessageId, cleanedResult, attachments);
					}
				} else if (createToolResultMessage) {
					toolResultMessage = await createToolResultMessage(
						toolCall.id,
						cleanedResult,
						attachments.length > 0 ? attachments : undefined
					);
				}

				if (attachments.length > 0 && toolResultMessage) {
					onAttachments?.(toolResultMessage.id, attachments);
				}

				// Build content parts for session history (including images for vision models)
				const contentParts: ApiChatMessageContentPart[] = [
					{ text: cleanedResult, type: ContentPartType.TEXT }
				];

				for (const attachment of attachments) {
					if (attachment.type === AttachmentType.AUDIO) {
						if (modelsStore.props.modelSupportsAudio(effectiveModel)) {
							contentParts.push({
								input_audio: {
									data: (attachment as DatabaseMessageExtraAudioFile).base64Data,
									format: getAudioInputFormat(
										(attachment as DatabaseMessageExtraAudioFile).mimeType
									)
								},
								type: ContentPartType.INPUT_AUDIO
							});
						}
					} else if (attachment.type === AttachmentType.IMAGE) {
						if (modelsStore.props.modelSupportsVision(effectiveModel)) {
							contentParts.push({
								image_url: {
									url: (attachment as DatabaseMessageExtraImageFile).base64Url
								},
								type: ContentPartType.IMAGE_URL
							});
						} else {
							console.info(
								`[AgenticStore] Skipping image attachment (model "${effectiveModel}" does not support vision)`
							);
						}
					}
				}

				sessionMessages.push({
					content: contentParts.length === 1 ? cleanedResult : contentParts,
					role: MessageRole.TOOL,
					tool_call_id: toolCall.id
				});
			}

			if (turnStats.toolCalls.length > 0) {
				agenticTimings.perTurn!.push(turnStats);

				const intermediateTimings = this.buildFinalTimings(capturedTimings, agenticTimings);

				if (intermediateTimings) onTurnComplete?.(intermediateTimings);
			}

			// If tools were interrupted by a steering message, exit now instead of starting another LLM turn
			if (this.gates.hasPendingSteeringMessage(conversationId)) {
				console.log(
					'[AgenticStore] Steering message detected after tool execution, exiting agentic flow'
				);
				onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));

				return;
			}

			turn++;
		}
	}

	private extractBase64Attachments(result: string): {
		cleanedResult: string;
		attachments: DatabaseMessageExtra[];
	} {
		if (!result.trim()) {
			return { attachments: [], cleanedResult: result };
		}

		const lines = result.split(NEWLINE);
		const attachments: DatabaseMessageExtra[] = [];

		let attachmentIndex = 0;

		const cleanedLines = lines.map((line) => {
			const trimmedLine = line.trim();
			const match = trimmedLine.match(DATA_URI_BASE64_REGEX);

			if (!match) {
				return line;
			}

			const mimeType = match[1].toLowerCase();
			const base64Data = match[2];

			if (!base64Data) {
				return line;
			}

			attachmentIndex += 1;
			const name = this.buildAttachmentName(mimeType, attachmentIndex);

			if (mimeType.startsWith(MIME_TYPE_PREFIXES.IMAGE)) {
				attachments.push({ base64Url: trimmedLine, name, type: AttachmentType.IMAGE });

				return `[Attachment saved: ${name}]`;
			}

			if (mimeType.startsWith(MimeTypePrefix.AUDIO)) {
				// audio extras hold the bare base64, the input_audio part has no room for a data URI
				attachments.push({
					base64Data,
					mimeType,
					name,
					type: AttachmentType.AUDIO
				});

				return `[Attachment saved: ${name}]`;
			}

			return line;
		});

		return { attachments, cleanedResult: cleanedLines.join(NEWLINE) };
	}

	private normalizeToolCalls(toolCalls: ApiChatCompletionToolCall[]): AgenticToolCallList {
		if (!toolCalls) return [];

		return toolCalls.map((call, index) => ({
			function: {
				arguments: call?.function?.arguments ?? '',
				name: call?.function?.name ?? ''
			},
			id: call?.id ?? `tool_${index}`,
			type: (call?.type as ToolCallType.FUNCTION) ?? ToolCallType.FUNCTION
		}));
	}

	private parseToolArguments(args: string | Record<string, unknown>): Record<string, unknown> {
		if (typeof args === 'object') return args;

		const trimmed = args.trim();

		if (trimmed === '') return {};

		return JSON.parse(trimmed) as Record<string, unknown>;
	}

	private updateSession(conversationId: string, update: Partial<AgenticSession>): void {
		const session = this.getSession(conversationId);

		this.sessions.set(conversationId, { ...session, ...update });
	}
}

export const agenticStore = new AgenticStore();
