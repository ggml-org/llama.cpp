/**
 * agenticStore - Reactive State Store for Agentic Loop Orchestration
 *
 * Manages multi-turn agentic loop with MCP tools:
 * - LLM streaming with tool call detection
 * - Tool execution via mcpStore
 * - Session state management
 * - Turn limit enforcement
 *
 * Each agentic turn produces separate DB messages:
 * - One assistant message per LLM turn (with tool_calls if any)
 * - One tool result message per tool call execution
 *
 * **Architecture & Relationships:**
 * - **ChatService**: Stateless API layer (sendMessage, streaming)
 * - **mcpStore**: MCP connection management and tool execution
 * - **agenticStore** (this): Reactive state + business logic
 *
 * @see ChatService in services/chat.service.ts for API operations
 * @see mcpStore in stores/mcp.svelte.ts for MCP operations
 */

import { ChatService } from '$lib/services';
import { config } from '$lib/stores/settings.svelte';
import { mcpStore } from '$lib/stores/mcp.svelte';
import { modelsStore } from '$lib/stores/models.svelte';
import { toolsStore } from '$lib/stores/tools.svelte';
import { permissionsStore } from '$lib/stores/permissions.svelte';
import { ToolSource, ToolPermissionDecision } from '$lib/enums';
import { SvelteMap } from 'svelte/reactivity';
import { ToolsService } from '$lib/services/tools.service';
import { isAbortError } from '$lib/utils';
import {
	DEFAULT_AGENTIC_CONFIG,
	NEWLINE_SEPARATOR,
	LLM_ERROR_BLOCK_START,
	LLM_ERROR_BLOCK_END
} from '$lib/constants';
import {
	IMAGE_MIME_TO_EXTENSION,
	DATA_URI_BASE64_REGEX,
	MCP_ATTACHMENT_NAME_PREFIX,
	DEFAULT_IMAGE_EXTENSION
} from '$lib/constants';
import {
	AttachmentType,
	ContentPartType,
	MessageRole,
	MimeTypePrefix,
	ToolCallType
} from '$lib/enums';
import type {
	AgenticFlowParams,
	AgenticFlowResult,
	AgenticSession,
	AgenticConfig,
	SettingsConfigType,
	McpServerOverride,
	MCPToolCall
} from '$lib/types';
import type {
	AgenticMessage,
	AgenticToolCallList,
	AgenticFlowCallbacks,
	AgenticFlowOptions,
	SteeringMessage
} from '$lib/types/agentic';
import type {
	ApiChatCompletionToolCall,
	ApiChatMessageData,
	ApiChatMessageContentPart
} from '$lib/types/api';
import type {
	ChatMessagePromptProgress,
	ChatMessageTimings,
	ChatMessageAgenticTimings,
	ChatMessageToolCallTiming,
	ChatMessageAgenticTurnStats
} from '$lib/types/chat';
import type {
	DatabaseMessage,
	DatabaseMessageExtra,
	DatabaseMessageExtraImageFile
} from '$lib/types/database';

function createDefaultSession(): AgenticSession {
	return {
		isRunning: false,
		currentTurn: 0,
		totalToolCalls: 0,
		lastError: null,
		streamingToolCall: null,
		pendingPermissionRequest: null
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
				role: MessageRole.ASSISTANT,
				content: message.content,
				reasoning_content: message.reasoning_content,
				tool_calls: message.tool_calls.map((call, index) => ({
					id: call.id ?? `call_${index}`,
					type: (call.type as ToolCallType.FUNCTION) ?? ToolCallType.FUNCTION,
					function: { name: call.function?.name ?? '', arguments: call.function?.arguments ?? '' }
				}))
			} satisfies AgenticMessage;
		}
		if (message.role === MessageRole.ASSISTANT) {
			return {
				role: MessageRole.ASSISTANT,
				content: message.content,
				reasoning_content: message.reasoning_content
			} satisfies AgenticMessage;
		}
		if (message.role === MessageRole.TOOL && message.tool_call_id) {
			return {
				role: MessageRole.TOOL,
				tool_call_id: message.tool_call_id,
				content: typeof message.content === 'string' ? message.content : ''
			} satisfies AgenticMessage;
		}
		return {
			role: message.role as MessageRole.SYSTEM | MessageRole.USER,
			content: message.content
		} satisfies AgenticMessage;
	});
}

class AgenticStore {
	private _sessions = new SvelteMap<string, AgenticSession>();
	/** Dedicated reactive state for pending permission requests (ensures immediate UI updates) */
	private _pendingPermissions = new SvelteMap<
		string,
		{ toolName: string; serverLabel: string } | null
	>();
	/** Non-reactive: stores resolve functions for pending permission Promises */
	private _permissionResolvers = new Map<string, (decision: ToolPermissionDecision) => void>();

	/** Dedicated reactive state for pending continue requests (turn limit reached) */
	private _pendingContinueRequests = new SvelteMap<string, boolean>();
	/** Non-reactive: stores resolve functions for pending continue Promises */
	private _continueResolvers = new Map<string, (shouldContinue: boolean) => void>();

	/** Reactive: queued steering messages to inject between turns */
	private _steeringMessages = new SvelteMap<string, SteeringMessage>();

	get isReady(): boolean {
		return true;
	}
	get isAnyRunning(): boolean {
		for (const session of this._sessions.values()) {
			if (session.isRunning) return true;
		}
		return false;
	}

	getSession(conversationId: string): AgenticSession {
		let session = this._sessions.get(conversationId);
		if (!session) {
			session = createDefaultSession();
			this._sessions.set(conversationId, session);
		}
		return session;
	}

	private updateSession(conversationId: string, update: Partial<AgenticSession>): void {
		const session = this.getSession(conversationId);
		this._sessions.set(conversationId, { ...session, ...update });
	}

	clearSession(conversationId: string): void {
		this._sessions.delete(conversationId);
	}

	getActiveSessions(): Array<{ conversationId: string; session: AgenticSession }> {
		const active: Array<{ conversationId: string; session: AgenticSession }> = [];
		for (const [conversationId, session] of this._sessions.entries()) {
			if (session.isRunning) active.push({ conversationId, session });
		}
		return active;
	}

	isRunning(conversationId: string): boolean {
		return this.getSession(conversationId).isRunning;
	}

	currentTurn(conversationId: string): number {
		return this.getSession(conversationId).currentTurn;
	}

	totalToolCalls(conversationId: string): number {
		return this.getSession(conversationId).totalToolCalls;
	}

	lastError(conversationId: string): Error | null {
		return this.getSession(conversationId).lastError;
	}

	streamingToolCall(conversationId: string): { name: string; arguments: string } | null {
		return this.getSession(conversationId).streamingToolCall;
	}

	pendingPermissionRequest(
		conversationId: string
	): { toolName: string; serverLabel: string } | null {
		return this._pendingPermissions.get(conversationId) ?? null;
	}

	pendingContinueRequest(conversationId: string): boolean {
		return this._pendingContinueRequests.get(conversationId) ?? false;
	}

	resolveContinue(conversationId: string, shouldContinue: boolean): void {
		const resolver = this._continueResolvers.get(conversationId);
		if (resolver) {
			this._continueResolvers.delete(conversationId);
			resolver(shouldContinue);
		}
	}

	resolvePermission(conversationId: string, decision: ToolPermissionDecision): void {
		const resolver = this._permissionResolvers.get(conversationId);
		if (resolver) {
			this._permissionResolvers.delete(conversationId);
			resolver(decision);
		}
	}

	clearError(conversationId: string): void {
		this.updateSession(conversationId, { lastError: null });
	}

	hasPendingSteeringMessage(conversationId: string): boolean {
		return this._steeringMessages.has(conversationId);
	}

	pendingSteeringMessageContent(conversationId: string): string | null {
		return this._steeringMessages.get(conversationId)?.content ?? null;
	}

	pendingSteeringMessageExtras(conversationId: string): DatabaseMessageExtra[] | undefined {
		return this._steeringMessages.get(conversationId)?.extras;
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
		this._steeringMessages.set(conversationId, { content, extras });
	}

	/**
	 * Clear the pending steering message without consuming it.
	 */
	clearSteeringMessage(conversationId: string): void {
		this._steeringMessages.delete(conversationId);
	}

	/**
	 * Consume and return the pending steering message for re-sending.
	 * Called by chatStore after the agentic flow exits.
	 */
	consumePendingSteeringMessage(conversationId: string): SteeringMessage | null {
		const msg = this._steeringMessages.get(conversationId);
		if (!msg) return null;
		this._steeringMessages.delete(conversationId);
		return msg;
	}

	getConfig(settings: SettingsConfigType, perChatOverrides?: McpServerOverride[]): AgenticConfig {
		const maxTurns = Number(settings.agenticMaxTurns) || DEFAULT_AGENTIC_CONFIG.maxTurns;
		const maxToolPreviewLines =
			Number(settings.agenticMaxToolPreviewLines) || DEFAULT_AGENTIC_CONFIG.maxToolPreviewLines;
		const hasTools =
			mcpStore.hasEnabledServers(perChatOverrides) ||
			toolsStore.builtinTools.length > 0 ||
			toolsStore.customTools.length > 0;
		return {
			enabled: hasTools && DEFAULT_AGENTIC_CONFIG.enabled,
			maxTurns,
			maxToolPreviewLines
		};
	}

	private parseToolArguments(args: string | Record<string, unknown>): Record<string, unknown> {
		if (typeof args === 'object') return args;
		const trimmed = args.trim();
		if (trimmed === '') return {};
		return JSON.parse(trimmed) as Record<string, unknown>;
	}

	private async requestPermission(
		conversationId: string,
		toolName: string,
		serverLabel: string,
		signal?: AbortSignal
	): Promise<ToolPermissionDecision> {
		const permissionKey = toolsStore.getPermissionKey(toolName);
		if (permissionKey && permissionsStore.hasTool(permissionKey)) {
			return ToolPermissionDecision.ONCE;
		}

		this._pendingPermissions.set(conversationId, { toolName, serverLabel });

		return new Promise<ToolPermissionDecision>((resolve) => {
			if (signal?.aborted) {
				this._pendingPermissions.set(conversationId, null);
				resolve(ToolPermissionDecision.DENY);
				return;
			}

			this._permissionResolvers.set(conversationId, (decision) => {
				this._pendingPermissions.set(conversationId, null);
				if (decision === ToolPermissionDecision.ALWAYS && permissionKey) {
					permissionsStore.allowTool(permissionKey);
				} else if (decision === ToolPermissionDecision.ALWAYS_SERVER) {
					const serverToolKeys = toolsStore.allTools
						.filter((t) =>
							t.serverName
								? t.serverName === serverLabel
								: toolsStore.getToolServerLabel(t.definition.function.name) === serverLabel
						)
						.map((t) => toolsStore.getPermissionKey(t.definition.function.name)!)
						.filter((k): k is string => k !== null);
					permissionsStore.allowTools(serverToolKeys);
				}
				resolve(decision);
			});

			signal?.addEventListener(
				'abort',
				() => {
					const resolver = this._permissionResolvers.get(conversationId);
					if (resolver) {
						this._permissionResolvers.delete(conversationId);
						this._pendingPermissions.set(conversationId, null);
						resolve(ToolPermissionDecision.DENY);
					}
				},
				{ once: true }
			);
		});
	}

	private async requestContinue(conversationId: string, signal?: AbortSignal): Promise<boolean> {
		this._pendingContinueRequests.set(conversationId, true);

		return new Promise<boolean>((resolve) => {
			if (signal?.aborted) {
				this._pendingContinueRequests.set(conversationId, false);
				resolve(false);
				return;
			}

			this._continueResolvers.set(conversationId, (shouldContinue) => {
				this._pendingContinueRequests.set(conversationId, false);
				resolve(shouldContinue);
			});

			signal?.addEventListener(
				'abort',
				() => {
					const resolver = this._continueResolvers.get(conversationId);
					if (resolver) {
						this._continueResolvers.delete(conversationId);
						this._pendingContinueRequests.set(conversationId, false);
						resolve(false);
					}
				},
				{ once: true }
			);
		});
	}

	async runAgenticFlow(params: AgenticFlowParams): Promise<AgenticFlowResult> {
		const { conversationId, messages, options = {}, callbacks, signal, perChatOverrides } = params;

		// Clear any pending permissions/continue requests for this conversation when starting a new flow
		this._pendingPermissions.set(conversationId, null);
		this._permissionResolvers.delete(conversationId);
		this._pendingContinueRequests.set(conversationId, false);
		this._continueResolvers.delete(conversationId);
		this._steeringMessages.delete(conversationId);

		// Ensure built-in tools are fetched before checking if agentic is enabled
		if (toolsStore.builtinTools.length === 0 && !toolsStore.loading) {
			await toolsStore.fetchBuiltinTools();
		}

		const agenticConfig = this.getConfig(config(), perChatOverrides);
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
			console.log('[AgenticStore] No tools available, falling back to standard chat');

			return { handled: false };
		}

		console.log(`[AgenticStore] Starting agentic flow with ${tools.length} tools`);

		const normalizedMessages: ApiChatMessageData[] = messages
			.map((msg) => {
				if ('id' in msg && 'convId' in msg && 'timestamp' in msg)
					return ChatService.convertDbMessageToApiChatMessageData(
						msg as DatabaseMessage & { extra?: DatabaseMessageExtra[] }
					);
				return msg as ApiChatMessageData;
			})
			.filter((msg) => {
				if (msg.role === MessageRole.SYSTEM) {
					const content = typeof msg.content === 'string' ? msg.content : '';
					return content.trim().length > 0;
				}
				return true;
			});

		this.updateSession(conversationId, {
			isRunning: true,
			currentTurn: 0,
			totalToolCalls: 0,
			lastError: null
		});

		if (hasMcpServers) mcpStore.acquireConnection();

		try {
			await this.executeAgenticLoop({
				conversationId,
				messages: normalizedMessages,
				options,
				tools,
				agenticConfig,
				callbacks,
				signal
			});
			return { handled: true };
		} catch (error) {
			const normalizedError = error instanceof Error ? error : new Error(String(error));
			this.updateSession(conversationId, { lastError: normalizedError });
			callbacks.onError?.(normalizedError);
			return { handled: true, error: normalizedError };
		} finally {
			this.updateSession(conversationId, { isRunning: false });

			if (hasMcpServers) {
				await mcpStore
					.releaseConnection()
					.catch((err: unknown) =>
						console.warn('[AgenticStore] Failed to release MCP connection:', err)
					);
			}
		}
	}

	private async executeAgenticLoop(params: {
		conversationId: string;
		messages: ApiChatMessageData[];
		options: AgenticFlowOptions;
		tools: ReturnType<typeof mcpStore.getToolDefinitionsForLLM>;
		agenticConfig: AgenticConfig;
		callbacks: AgenticFlowCallbacks;
		signal?: AbortSignal;
	}): Promise<void> {
		const { conversationId, messages, options, tools, agenticConfig, callbacks, signal } = params;
		const {
			onChunk,
			onReasoningChunk,
			onToolCallsStreaming,
			onAttachments,
			onModel,
			onAssistantTurnComplete,
			createToolResultMessage,
			createAssistantMessage,
			onFlowComplete,
			onTimings,
			onTurnComplete
		} = callbacks;

		const sessionMessages: AgenticMessage[] = toAgenticMessages(messages);
		let capturedTimings: ChatMessageTimings | undefined;
		let totalToolCallCount = 0;

		const agenticTimings: ChatMessageAgenticTimings = {
			turns: 0,
			toolCallsCount: 0,
			toolsMs: 0,
			toolCalls: [],
			perTurn: [],
			llm: { predicted_n: 0, predicted_ms: 0, prompt_n: 0, prompt_ms: 0 }
		};
		const maxTurns = agenticConfig.maxTurns;

		const effectiveModel = options.model || modelsStore.models[0]?.model || '';

		let turn = 0;
		while (true) {
			if (turn >= maxTurns) {
				// Turn limit reached - ask user whether to continue
				const shouldContinue = await this.requestContinue(conversationId, signal);

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
				turn: turn + 1,
				llm: { predicted_n: 0, predicted_ms: 0, prompt_n: 0, prompt_ms: 0 },
				toolCalls: [],
				toolsMs: 0
			};

			try {
				await ChatService.sendMessage(
					sessionMessages as ApiChatMessageData[],
					{
						...options,
						stream: true,
						tools: tools.length > 0 ? tools : undefined,
						onChunk: (chunk: string) => {
							turnContent += chunk;
							onChunk?.(chunk);
						},
						onReasoningChunk: (chunk: string) => {
							turnReasoningContent += chunk;
							onReasoningChunk?.(chunk);
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
											streamingToolCall: { name, arguments: args }
										});
									}
								}
							} catch {
								/* Ignore parse errors during streaming */
							}
						},
						onModel,
						onTimings: (timings?: ChatMessageTimings, progress?: ChatMessagePromptProgress) => {
							onTimings?.(timings, progress);
							if (timings) {
								capturedTimings = timings;
								turnTimings = timings;
							}
						},
						onComplete: () => {
							/* Completion handled after sendMessage resolves */
						},
						onError: (error: Error) => {
							throw error;
						}
					},
					undefined,
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
				// Save error as content in the current turn
				onChunk?.(`${LLM_ERROR_BLOCK_START}${normalizedError.message}${LLM_ERROR_BLOCK_END}`);
				await onAssistantTurnComplete?.(
					turnContent + `${LLM_ERROR_BLOCK_START}${normalizedError.message}${LLM_ERROR_BLOCK_END}`,
					turnReasoningContent || undefined,
					this.buildFinalTimings(capturedTimings, agenticTimings),
					undefined
				);
				onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));
				throw normalizedError;
			}

			// === Steering check: if a user message was queued during this turn, exit the flow.
			// The caller (chatStore) will consume the pending message and re-send it normally.
			if (this._steeringMessages.has(conversationId)) {
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
				role: MessageRole.ASSISTANT,
				content: turnContent || undefined,
				reasoning_content: turnReasoningContent || undefined,
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
				if (this._steeringMessages.has(conversationId)) {
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
							role: MessageRole.TOOL,
							tool_call_id: remainingCall.id,
							content: interruptedContent
						});
					}
					break;
				}

				const toolName = toolCall.function.name;
				const serverLabel = toolsStore.getToolServerLabel(toolName);

				// Ask for permission before executing the tool
				const permission = await this.requestPermission(
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

				let result: string;
				let toolSuccess = true;

				if (permission === ToolPermissionDecision.DENY) {
					result = 'Tool execution was denied by the user.';
					toolSuccess = false;
				} else {
					try {
						if (toolSource === ToolSource.BUILTIN) {
							const args = this.parseToolArguments(toolCall.function.arguments);
							const executionResult = await ToolsService.executeTool(toolName, args, signal);

							result = executionResult.content;

							if (executionResult.isError) toolSuccess = false;
						} else {
							const mcpCall: MCPToolCall = {
								id: toolCall.id,
								function: { name: toolName, arguments: toolCall.function.arguments }
							};
							const executionResult = await mcpStore.executeTool(mcpCall, signal);

							result = executionResult.content;
						}
					} catch (error) {
						if (isAbortError(error)) {
							onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));
							return;
						}
						result = `Error: ${error instanceof Error ? error.message : String(error)}`;
						toolSuccess = false;
					}
				}

				const toolDurationMs = performance.now() - toolStartTime;
				const toolTiming: ChatMessageToolCallTiming = {
					name: toolCall.function.name,
					duration_ms: Math.round(toolDurationMs),
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

				const { cleanedResult, attachments } = this.extractBase64Attachments(result);

				// Create the tool result message in the DB
				let toolResultMessage: DatabaseMessage | undefined;
				if (createToolResultMessage) {
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
					{ type: ContentPartType.TEXT, text: cleanedResult }
				];
				for (const attachment of attachments) {
					if (attachment.type === AttachmentType.IMAGE) {
						if (modelsStore.modelSupportsVision(effectiveModel)) {
							contentParts.push({
								type: ContentPartType.IMAGE_URL,
								image_url: { url: (attachment as DatabaseMessageExtraImageFile).base64Url }
							});
						} else {
							console.info(
								`[AgenticStore] Skipping image attachment (model "${effectiveModel}" does not support vision)`
							);
						}
					}
				}

				sessionMessages.push({
					role: MessageRole.TOOL,
					tool_call_id: toolCall.id,
					content: contentParts.length === 1 ? cleanedResult : contentParts
				});
			}

			if (turnStats.toolCalls.length > 0) {
				agenticTimings.perTurn!.push(turnStats);

				const intermediateTimings = this.buildFinalTimings(capturedTimings, agenticTimings);
				if (intermediateTimings) onTurnComplete?.(intermediateTimings);
			}

			// If tools were interrupted by a steering message, exit now instead of starting another LLM turn
			if (this._steeringMessages.has(conversationId)) {
				console.log(
					'[AgenticStore] Steering message detected after tool execution, exiting agentic flow'
				);
				onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));
				return;
			}

			turn++;
		}
	}

	private buildFinalTimings(
		capturedTimings: ChatMessageTimings | undefined,
		agenticTimings: ChatMessageAgenticTimings
	): ChatMessageTimings | undefined {
		if (agenticTimings.toolCallsCount === 0) return capturedTimings;
		return {
			predicted_n: capturedTimings?.predicted_n,
			predicted_ms: capturedTimings?.predicted_ms,
			prompt_n: capturedTimings?.prompt_n,
			prompt_ms: capturedTimings?.prompt_ms,
			cache_n: capturedTimings?.cache_n,
			agentic: agenticTimings
		};
	}

	private normalizeToolCalls(toolCalls: ApiChatCompletionToolCall[]): AgenticToolCallList {
		if (!toolCalls) return [];
		return toolCalls.map((call, index) => ({
			id: call?.id ?? `tool_${index}`,
			type: (call?.type as ToolCallType.FUNCTION) ?? ToolCallType.FUNCTION,
			function: { name: call?.function?.name ?? '', arguments: call?.function?.arguments ?? '' }
		}));
	}

	private extractBase64Attachments(result: string): {
		cleanedResult: string;
		attachments: DatabaseMessageExtra[];
	} {
		if (!result.trim()) {
			return { cleanedResult: result, attachments: [] };
		}

		const lines = result.split(NEWLINE_SEPARATOR);
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

			if (mimeType.startsWith(MimeTypePrefix.IMAGE)) {
				attachments.push({ type: AttachmentType.IMAGE, name, base64Url: trimmedLine });

				return `[Attachment saved: ${name}]`;
			}

			return line;
		});

		return { cleanedResult: cleanedLines.join(NEWLINE_SEPARATOR), attachments };
	}

	private buildAttachmentName(mimeType: string, index: number): string {
		const extension = IMAGE_MIME_TO_EXTENSION[mimeType] ?? DEFAULT_IMAGE_EXTENSION;

		return `${MCP_ATTACHMENT_NAME_PREFIX}-${Date.now()}-${index}.${extension}`;
	}
}

export const agenticStore = new AgenticStore();

export function agenticIsRunning(conversationId: string) {
	return agenticStore.isRunning(conversationId);
}

export function agenticCurrentTurn(conversationId: string) {
	return agenticStore.currentTurn(conversationId);
}

export function agenticTotalToolCalls(conversationId: string) {
	return agenticStore.totalToolCalls(conversationId);
}

export function agenticLastError(conversationId: string) {
	return agenticStore.lastError(conversationId);
}

export function agenticStreamingToolCall(conversationId: string) {
	return agenticStore.streamingToolCall(conversationId);
}

export function agenticPendingPermissionRequest(conversationId: string) {
	return agenticStore.pendingPermissionRequest(conversationId);
}

export function agenticResolvePermission(conversationId: string, decision: ToolPermissionDecision) {
	agenticStore.resolvePermission(conversationId, decision);
}

export function agenticPendingContinueRequest(conversationId: string) {
	return agenticStore.pendingContinueRequest(conversationId);
}

export function agenticResolveContinue(conversationId: string, shouldContinue: boolean) {
	agenticStore.resolveContinue(conversationId, shouldContinue);
}

export function agenticHasPendingSteeringMessage(conversationId: string) {
	return agenticStore.hasPendingSteeringMessage(conversationId);
}

export function agenticInjectSteeringMessage(
	conversationId: string,
	content: string,
	extras?: DatabaseMessageExtra[]
) {
	agenticStore.injectSteeringMessage(conversationId, content, extras);
}

export function agenticPendingSteeringMessageContent(conversationId: string) {
	return agenticStore.pendingSteeringMessageContent(conversationId);
}

export function agenticPendingSteeringMessageExtras(conversationId: string) {
	return agenticStore.pendingSteeringMessageExtras(conversationId);
}

export function agenticClearSteeringMessage(conversationId: string) {
	agenticStore.clearSteeringMessage(conversationId);
}

export function agenticIsAnyRunning() {
	return agenticStore.isAnyRunning;
}
