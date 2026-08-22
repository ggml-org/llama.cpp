/**
 * ConversationPreferences - Per-chat options with global fallback
 *
 * Owns the options that resolve per conversation: the tool policy (disabled
 * categories and tool keys), reasoning effort, and the working directory.
 * Tool picks made on the empty new-chat screen edit the global defaults
 * directly (they seed every newly created conversation); cwd and reasoning
 * effort are buffered as pending state and threaded into the next created
 * conversation by the host.
 * Created and owned by conversationsStore; the host owns the conversation
 * rows these options persist onto.
 */

import { REASONING_EFFORT_DEFAULT_LOCALSTORAGE_KEY } from '$lib/constants';
import { ReasoningEffort, ToolSource } from '$lib/enums';
import { DatabaseService } from '$lib/services/database.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { toolsStore } from '$lib/stores/tools.svelte';
import type { DatabaseConversation, ToolEntry, ToolGroup } from '$lib/types';

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

/** Effective disabled tool keys: conversation row, falling back to defaults. */
function buildDisabledTools(conv: DatabaseConversation | null): Set<string> {
	return new Set(conv ? (conv.disabledTools ?? []) : [...toolsStore.disabledTools]);
}

/** Effective disabled tool categories: conversation row, falling back to defaults. */
function buildDisabledToolCategories(conv: DatabaseConversation | null): Set<ToolSource> {
	return new Set(
		conv ? (conv.disabledToolCategories ?? []) : [...toolsStore.disabledToolCategories]
	);
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

	private get _disabledToolCategories(): Set<ToolSource> {
		return buildDisabledToolCategories(this.host.activeConversation);
	}

	// getters, not $derived fields: lazy evaluation keeps them off the class
	// field initialization order (host is assigned by the constructor), and
	// reads of the underlying $state stay tracked in reactive contexts
	private get _disabledTools(): Set<string> {
		return buildDisabledTools(this.host.activeConversation);
	}

	/**
	 *
	 *
	 * Tool Policy
	 *
	 *
	 */

	constructor(private host: ConversationsPreferencesHost) {}

	/** Effective disabled tool categories for the current context, captured at flow start. */
	getDisabledToolCategories(): ToolSource[] {
		return [...this._disabledToolCategories];
	}

	/** Effective disabled tool keys for the current context, captured at flow start. */
	getDisabledTools(): string[] {
		return [...this._disabledTools];
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

	/** Defaults snapshot for seeding a newly created conversation. */
	getToolPolicySnapshot(): { disabledTools?: string[]; disabledToolCategories?: ToolSource[] } {
		const disabledTools = [...toolsStore.disabledTools];
		const disabledToolCategories = [...toolsStore.disabledToolCategories];

		return {
			disabledToolCategories: disabledToolCategories.length ? disabledToolCategories : undefined,
			disabledTools: disabledTools.length ? disabledTools : undefined
		};
	}

	hasEnabledCwdTools(): boolean {
		return toolsStore.hasEnabledCwdTools(this._disabledTools, this._disabledToolCategories);
	}

	isCategoryEnabled(source: ToolSource): boolean {
		return !this._disabledToolCategories.has(source);
	}

	/** Group checkbox state: the category flag, or the server key for MCP groups. */
	isGroupChecked(group: ToolGroup): boolean {
		return group.source === ToolSource.MCP && group.serverId
			? this.isServerToolsEnabled(group.serverId)
			: this.isCategoryEnabled(group.source);
	}

	/** Server-scoped MCP group state: one key disables all of that server's tools. */
	isServerToolsEnabled(serverId: string): boolean {
		return this.isToolEnabled(toolsStore.getMcpServerToolsKey(serverId));
	}

	/** Effective state: own key, MCP server group key, and category all on. */
	isToolActive(entry: ToolEntry): boolean {
		return toolsStore.isEntryEnabled(entry, this._disabledTools, this._disabledToolCategories);
	}

	/** Own-level state: the tool key itself, ignoring category and server group. */
	isToolEnabled(key: string): boolean {
		return !this._disabledTools.has(key);
	}

	/** True when a parent level (category or MCP server group) disables this entry. */
	isToolParentDisabled(entry: ToolEntry): boolean {
		if (!this.isCategoryEnabled(entry.source)) return true;

		return (
			entry.source === ToolSource.MCP &&
			!!entry.serverId &&
			!this.isServerToolsEnabled(entry.serverId)
		);
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

		this.host.applyConversationUpdate(this.host.activeConversation.id, {
			reasoningEffort: effort
		});

		await DatabaseService.updateConversation(this.host.activeConversation.id, {
			reasoningEffort: effort
		});
	}

	async toggleCategory(source: ToolSource): Promise<void> {
		const conv: DatabaseConversation | null = this.host.activeConversation;

		if (!conv) {
			toolsStore.toggleCategory(source);

			return;
		}

		const next = buildDisabledToolCategories(conv);

		if (next.has(source)) next.delete(source);
		else next.add(source);

		await this.persistDisabledToolCategories(next);
	}

	async toggleGroup(group: ToolGroup): Promise<void> {
		if (group.source === ToolSource.MCP && group.serverId) {
			await this.toggleServerTools(group.serverId);
		} else {
			await this.toggleCategory(group.source);
		}
	}

	async toggleServerTools(serverId: string): Promise<void> {
		await this.toggleTool(toolsStore.getMcpServerToolsKey(serverId));
	}

	/**
	 *
	 *
	 * Reasoning Effort
	 *
	 *
	 */

	async toggleTool(key: string): Promise<void> {
		const conv: DatabaseConversation | null = this.host.activeConversation;

		if (!conv) {
			toolsStore.toggleTool(key);

			return;
		}

		const next = buildDisabledTools(conv);

		if (next.has(key)) next.delete(key);
		else next.add(key);

		await this.persistDisabledTools(next);
	}

	private async persistDisabledToolCategories(disabled: Set<ToolSource>): Promise<void> {
		const conv = this.host.activeConversation;

		if (!conv) return;

		const disabledToolCategories = disabled.size ? [...disabled] : undefined;

		this.host.applyConversationUpdate(conv.id, { disabledToolCategories });

		await DatabaseService.updateConversation(conv.id, { disabledToolCategories });
	}

	/**
	 *
	 *
	 * Working Directory
	 *
	 *
	 */

	private async persistDisabledTools(disabled: Set<string>): Promise<void> {
		const conv = this.host.activeConversation;

		if (!conv) return;

		const disabledTools = disabled.size ? [...disabled] : undefined;

		this.host.applyConversationUpdate(conv.id, { disabledTools });

		await DatabaseService.updateConversation(conv.id, { disabledTools });
	}
}
