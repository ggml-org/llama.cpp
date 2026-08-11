/**
 * STORES
 *
 * Reactive Svelte runes state layer. Stores own application state and
 * expose it as plain Svelte 5 runes (`$state`, `$derived`, `$effect`),
 * consumed by components, routes, hooks and services.
 *
 * Import from this barrel in leaf consumers:
 *
 * ```ts
 * import { chatStore, modelsStore, selectedModelName } from '$lib/stores';
 * ```
 *
 * Store modules keep direct imports between each other (and from services/
 * utils they depend on) to avoid circular dependency chains.
 *
 * Each store below documents its primary responsibility.
 */

// CHAT / MESSAGING
export {
	chatStore,
	activeProcessingState,
	currentResponse,
	errorDialog,
	getAddFilesHandler,
	getAllLoadingChats,
	getAllStreamingChats,
	getChatStreaming,
	isChatLoading,
	isChatStreaming,
	isEditing,
	isLoading,
	isReasoning,
	pendingEditMessageId,
	chatHasPendingMessage,
	chatPendingMessageContent,
	chatPendingMessageExtras,
	chatClearPendingMessage,
	chatInjectPendingMessage
} from './chat.svelte';

export { draftMessagesStore } from './draft-messages.svelte';

// AGENTIC (multi-turn tool orchestration)
export { agenticStore } from './agentic.svelte';

// CONVERSATIONS
export type { ConversationTreeItem } from './conversations.svelte';
export {
	conversationsStore,
	conversations,
	activeConversation,
	activeMessages,
	pendingCwd,
	isConversationsInitialized,
	buildConversationTree
} from './conversations.svelte';

// MCP
export {
	mcpStore,
	mcpIsInitializing,
	mcpIsInitialized,
	mcpError,
	mcpIsEnabled,
	mcpIsProxyAvailable,
	mcpAvailableTools,
	mcpConnectedServerCount,
	mcpConnectedServerNames,
	mcpToolCount,
	mcpServerInstructions,
	mcpHasServerInstructions,
	mcpHasResourcesCapability,
	mcpServersWithResources,
	mcpResourceContext
} from './mcp.svelte';

export {
	mcpResourceStore,
	mcpResources,
	mcpResourceAttachments,
	mcpResourceAttachmentCount,
	mcpHasResourceAttachments,
	mcpTotalResourceCount,
	mcpResourcesLoading
} from './mcp-resources.svelte';

// MODELS
export {
	modelsStore,
	modelOptions,
	routerModels,
	modelsLoading,
	modelsUpdating,
	modelsError,
	selectedModelId,
	selectedModelName,
	selectedModelOption,
	loadedModelIds,
	loadingModelIds,
	propsCacheVersion,
	singleModelName,
	selectedModelContextSize,
	favoriteModelIds,
	supportsThinking,
	checkModelSupportsThinking,
	thinkingSupportDetails
} from './models.svelte';

// SERVER
export {
	serverStore,
	serverProps,
	serverLoading,
	serverError,
	serverStatus,
	serverRole,
	defaultParams,
	contextSize,
	isRouterMode,
	isModelMode
} from './server.svelte';

// SETTINGS / UI PREFERENCES
export { settingsStore, config, isInitialized } from './settings.svelte';

export { settingsReferrer } from './settings-referrer.svelte';

export { permissionsStore } from './permissions.svelte';

// TOOLS
export { toolsStore, allTools, allToolDefinitions, toolGroups } from './tools.svelte';

// ENVIRONMENT / META
export { buildInfoStore } from './build-info.svelte';

export { versionStore } from './version.svelte';

export { device } from './device.svelte';

export { viewport, isMobile } from './viewport.svelte';

// shared via theme.svelte (reactive system theme detection) - settings.svelte
// also exports an unused `theme` getter which is intentionally NOT re-exported
export { theme } from './theme.svelte';

export {
	gaugePopup,
	gaugePopupClose,
	gaugeTriggerPointerDown,
	gaugeTriggerClick,
	gaugeTriggerKeydown,
	gaugeTriggerEnter,
	gaugeTriggerLeave,
	gaugeCardEnter,
	gaugeCardLeave
} from './context-gauge-popup.svelte';

export { persisted } from './persisted.svelte';
