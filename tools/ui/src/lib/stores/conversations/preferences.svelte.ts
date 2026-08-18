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
import type { conversationsStore } from '$lib/stores/conversations/index.svelte';
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
	/** Global (non-conversation-specific) reasoning effort default */
	pendingReasoningEffort = $state<ReasoningEffort>(loadReasoningEffortDefault());

	/**
	 * Working directory picked on the empty new-chat screen, before any
	 * conversation exists. Consumed by `chatStore.sendMessage()`, which
	 * records it into chat history as a synthetic message on first send.
	 * Cleared by `loadConversation` and `clearActiveConversation` so a
	 * stale pick can't bleed onto an unrelated chat.
	 */
	pendingCwd = $state<string | null>(null);

	constructor(private host: typeof conversationsStore) {}

	/** Reload persisted defaults, e.g. when the active conversation is cleared. */
	resetPending(): void {
		this.pendingReasoningEffort = loadReasoningEffortDefault();
		this.pendingCwd = null;
	}

	/**
	 *
	 *
	 * Tool Policy
	 *
	 *
	 */

	// getters, not $derived fields: lazy evaluation keeps them off the class
	// field initialization order (host is assigned by the constructor), and
	// reads of the underlying $state stay tracked in reactive contexts
	private get _disabledTools(): Set<string> {
		return buildDisabledTools(this.host.activeConversation);
	}

	private get _disabledToolCategories(): Set<ToolSource> {
		return buildDisabledToolCategories(this.host.activeConversation);
	}

	/** Effective disabled tool keys for the current context, captured at flow start. */
	getDisabledTools(): string[] {
		return [...this._disabledTools];
	}

	/** Effective disabled tool categories for the current context, captured at flow start. */
	getDisabledToolCategories(): ToolSource[] {
		return [...this._disabledToolCategories];
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

	/** Own-level state: the tool key itself, ignoring category and server group. */
	isToolEnabled(key: string): boolean {
		return !this._disabledTools.has(key);
	}

	/** Effective state: own key, MCP server group key, and category all on. */
	isToolActive(entry: ToolEntry): boolean {
		return toolsStore.isEntryEnabled(entry, this._disabledTools, this._disabledToolCategories);
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

	isCategoryEnabled(source: ToolSource): boolean {
		return !this._disabledToolCategories.has(source);
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

	/** Server-scoped MCP group state: one key disables all of that server's tools. */
	isServerToolsEnabled(serverId: string): boolean {
		return this.isToolEnabled(toolsStore.getMcpServerToolsKey(serverId));
	}

	async toggleServerTools(serverId: string): Promise<void> {
		await this.toggleTool(toolsStore.getMcpServerToolsKey(serverId));
	}

	/** Group checkbox state: the category flag, or the server key for MCP groups. */
	isGroupChecked(group: ToolGroup): boolean {
		return group.source === ToolSource.MCP && group.serverId
			? this.isServerToolsEnabled(group.serverId)
			: this.isCategoryEnabled(group.source);
	}

	async toggleGroup(group: ToolGroup): Promise<void> {
		if (group.source === ToolSource.MCP && group.serverId) {
			await this.toggleServerTools(group.serverId);
		} else {
			await this.toggleCategory(group.source);
		}
	}

	hasEnabledCwdTools(): boolean {
		return toolsStore.hasEnabledCwdTools(this._disabledTools, this._disabledToolCategories);
	}

	private async persistDisabledTools(disabled: Set<string>): Promise<void> {
		const conv = this.host.activeConversation;

		if (!conv) return;

		const disabledTools = disabled.size ? [...disabled] : undefined;

		await DatabaseService.updateConversation(conv.id, { disabledTools });

		this.host.activeConversation = { ...conv, disabledTools };

		const convIndex = this.host.conversations.findIndex((c) => c.id === conv.id);

		if (convIndex !== -1) {
			this.host.conversations[convIndex].disabledTools = disabledTools;
		}
	}

	private async persistDisabledToolCategories(disabled: Set<ToolSource>): Promise<void> {
		const conv = this.host.activeConversation;

		if (!conv) return;

		const disabledToolCategories = disabled.size ? [...disabled] : undefined;

		await DatabaseService.updateConversation(conv.id, { disabledToolCategories });

		this.host.activeConversation = { ...conv, disabledToolCategories };

		const convIndex = this.host.conversations.findIndex((c) => c.id === conv.id);

		if (convIndex !== -1) {
			this.host.conversations[convIndex].disabledToolCategories = disabledToolCategories;
		}
	}

	/**
	 *
	 *
	 * Reasoning Effort
	 *
	 *
	 */

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

		this.host.activeConversation = {
			...this.host.activeConversation,
			reasoningEffort: effort
		};

		await DatabaseService.updateConversation(this.host.activeConversation.id, {
			reasoningEffort: effort
		});

		const convIndex = this.host.conversations.findIndex(
			(c) => c.id === this.host.activeConversation!.id
		);

		if (convIndex !== -1) {
			this.host.conversations[convIndex].reasoningEffort = effort;
		}
	}

	/**
	 *
	 *
	 * Working Directory
	 *
	 *
	 */

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

		this.host.activeConversation = {
			...this.host.activeConversation,
			cwd: trimmed
		};

		await DatabaseService.updateConversation(this.host.activeConversation.id, {
			cwd: trimmed
		});

		const convIndex = this.host.conversations.findIndex(
			(c) => c.id === this.host.activeConversation!.id
		);

		if (convIndex !== -1) {
			this.host.conversations[convIndex].cwd = trimmed;
			this.host.conversations = [...this.host.conversations];
		}

		this.pendingCwd = null;
	}
}
