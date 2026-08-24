/**
 * ConversationPreferences - Per-chat options with global fallback
 *
 * Owns the options that resolve per conversation: MCP server overrides,
 * reasoning effort, and the working directory. Cwd and reasoning effort are
 * buffered as pending state and threaded into the next created conversation
 * by the host; MCP server overrides edit the sparse `mcpServerOverrides`
 * list on the active row (new-chat toggles edit the server's global flag).
 * Created and owned by conversationsStore; the host owns the conversation
 * rows these options persist onto.
 */

import { REASONING_EFFORT_DEFAULT_LOCALSTORAGE_KEY } from '$lib/constants';
import { ReasoningEffort } from '$lib/enums';
import { DatabaseService } from '$lib/services/database.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { mcpStore } from '$lib/stores/mcp/index.svelte';
import type { McpServerOverride } from '$lib/types/database';

/** Load reasoning effort default from localStorage, DEFAULT defers to the server */
function loadReasoningEffortDefault(): ReasoningEffort {
	if (typeof globalThis.localStorage === 'undefined') return ReasoningEffort.DEFAULT;

	try {
		const raw = localStorage.getItem(REASONING_EFFORT_DEFAULT_LOCALSTORAGE_KEY);

		return (raw as ReasoningEffort) || ReasoningEffort.DEFAULT;
	} catch {
		return ReasoningEffort.DEFAULT;
	}
}

/** Persist reasoning effort default to localStorage */
function saveReasoningEffortDefault(effort: ReasoningEffort): void {
	if (typeof globalThis.localStorage === 'undefined') return;

	localStorage.setItem(REASONING_EFFORT_DEFAULT_LOCALSTORAGE_KEY, effort);
}

/**
 * The slice of conversationsStore the preferences read and write. Kept narrow
 * on purpose so they cannot reach around the host's full surface;
 * conversationsStore implements this structurally.
 */
export interface ConversationsPreferencesHost {
	activeConversation: DatabaseConversation | null;
	conversations: DatabaseConversation[];
	applyConversationUpdate(id: string, updates: Partial<DatabaseConversation>): void;
}

export class ConversationPreferences {
	/**
	 * Working directory picked on the empty new-chat screen, before any
	 * conversation exists. Consumed by `chatStore.sendMessage()`, which
	 * records it into chat history as a synthetic message on first send.
	 * Cleared by `loadConversation` and `clearActiveConversation` so a
	 * stale pick can't bleed onto an unrelated chat.
	 */
	pendingCwd = $state<string | null>(null);

	/** Global (non-conversation-specific) reasoning effort default */
	pendingReasoningEffort = $state<ReasoningEffort>(loadReasoningEffortDefault());

	constructor(private host: ConversationsPreferencesHost) {}

	/**
	 * Gets the effective override list for the current conversation:
	 * one entry per configured server, resolved per server. The stored
	 * per-conversation list is sparse and only holds explicit toggles.
	 */
	getAllMcpServerOverrides(): McpServerOverride[] {
		const overrides = this.host.activeConversation?.mcpServerOverrides;

		return mcpStore.getServers().map((s) => {
			const override = overrides?.find((o: McpServerOverride) => o.serverId === s.id);

			return { enabled: override?.enabled ?? s.enabled, serverId: s.id };
		});
	}

	/**
	 * Gets the effective MCP server override for a specific server.
	 * A per-conversation override wins when present; a server without one
	 * resolves to its `mcpServers[i].enabled` default.
	 */
	getMcpServerOverride(serverId: string): McpServerOverride | undefined {
		const override = this.host.activeConversation?.mcpServerOverrides?.find(
			(o: McpServerOverride) => o.serverId === serverId
		);

		if (override) return override;

		return this.getDefaultOverride(serverId);
	}

	/**
	 * Gets the effective reasoning effort for the active conversation.
	 * Returns the conversation override if set, otherwise the global default.
	 * DEFAULT means no override is sent and the server decides.
	 */
	getReasoningEffort(): ReasoningEffort {
		if (this.host.activeConversation) {
			if (this.host.activeConversation.reasoningEffort !== undefined) {
				return this.host.activeConversation.reasoningEffort;
			}

			// conversations created before the tri-state store an explicit
			// opt-out only as thinkingEnabled = false
			if (this.host.activeConversation.thinkingEnabled === false) {
				return ReasoningEffort.OFF;
			}
		}

		return this.pendingReasoningEffort;
	}

	/** Checks if an MCP server is enabled for the active conversation. */
	isMcpServerEnabledForChat(serverId: string): boolean {
		const override = this.getMcpServerOverride(serverId);

		return override?.enabled ?? false;
	}

	/** Removes MCP server override for the active conversation. */
	async removeMcpServerOverride(serverId: string): Promise<void> {
		await this.setMcpServerOverride(serverId, undefined);
	}

	/** Reload persisted defaults, e.g. when the active conversation is cleared. */
	resetPending(): void {
		this.pendingReasoningEffort = loadReasoningEffortDefault();
		this.pendingCwd = null;
	}

	/**
	 * Sets the working directory for the active conversation. Pass `null` or
	 * an empty string to clear it, which restores the picker's empty state.
	 *
	 * On the empty new-chat screen (no active conversation yet), the value
	 * is buffered into `pendingCwd` so the user can pick before
	 * sending the first message; `createConversation()` consumes it.
	 *
	 * @param value - Absolute server-side path to the working directory, or null to clear
	 */
	async setCwd(value: string | null): Promise<void> {
		const trimmed = value?.trim() || undefined;

		// No chat yet - buffer for the first chat the user creates.
		if (!this.host.activeConversation) {
			this.pendingCwd = trimmed ?? null;

			return;
		}

		const id = this.host.activeConversation.id;

		this.host.applyConversationUpdate(id, {
			cwd: trimmed
		});

		await DatabaseService.updateConversation(id, {
			cwd: trimmed
		});

		this.pendingCwd = null;
	}

	/**
	 * Sets or removes MCP server override for the active conversation.
	 * If no conversation exists, persists `enabled` onto `mcpServers[i].enabled`
	 * (the single source of truth for new-chat defaults).
	 */
	async setMcpServerOverride(serverId: string, enabled: boolean | undefined): Promise<void> {
		if (!this.host.activeConversation) {
			if (enabled !== undefined) {
				mcpStore.updateServer(serverId, { enabled });
			}

			return;
		}

		// Clone to plain objects to avoid Proxy serialization issues with IndexedDB
		const currentOverrides = (this.host.activeConversation.mcpServerOverrides || []).map(
			(o: McpServerOverride) => ({
				enabled: o.enabled,
				serverId: o.serverId
			})
		);

		let newOverrides: McpServerOverride[];

		if (enabled === undefined) {
			newOverrides = currentOverrides.filter((o: McpServerOverride) => o.serverId !== serverId);
		} else {
			const existingIndex = currentOverrides.findIndex(
				(o: McpServerOverride) => o.serverId === serverId
			);

			if (existingIndex >= 0) {
				newOverrides = [...currentOverrides];
				newOverrides[existingIndex] = { enabled, serverId };
			} else {
				newOverrides = [...currentOverrides, { enabled, serverId }];
			}
		}

		const overrides = newOverrides.length > 0 ? newOverrides : undefined;
		const id = this.host.activeConversation.id;

		this.host.applyConversationUpdate(id, {
			mcpServerOverrides: overrides
		});

		await DatabaseService.updateConversation(id, {
			mcpServerOverrides: overrides
		});
	}

	/**
	 * Sets the reasoning effort for the active conversation.
	 * If no conversation exists, stores the global default.
	 * @param effort - The effort level ('default' | 'off' | 'low' | 'medium' | 'high' | 'max')
	 */
	async setReasoningEffort(effort: ReasoningEffort): Promise<void> {
		if (!this.host.activeConversation) {
			this.pendingReasoningEffort = effort;
			saveReasoningEffortDefault(effort);

			return;
		}

		const id = this.host.activeConversation.id;

		this.host.applyConversationUpdate(id, {
			reasoningEffort: effort
		});

		await DatabaseService.updateConversation(id, {
			reasoningEffort: effort
		});
	}

	/** Toggles MCP server enabled state for the active conversation. */
	async toggleMcpServerForChat(serverId: string): Promise<void> {
		const currentEnabled = this.isMcpServerEnabledForChat(serverId);

		await this.setMcpServerOverride(serverId, !currentEnabled);
	}

	/**
	 * Resolve the default enabled value for a server: its own `enabled`
	 * flag in `mcpServers`, so the global on/off state lives in one place.
	 */
	private getDefaultOverride(serverId: string): McpServerOverride | undefined {
		const server = mcpStore.getServers().find((s) => s.id === serverId);

		if (!server) return undefined;

		return { enabled: server.enabled, serverId };
	}
}
