import type { ChatMessageTimings, ChatRole, ChatMessageType } from '$lib/types/chat';
import { AttachmentType } from '$lib/enums';

/**
 * Where a skill lives.
 *
 *   - `lib`     — created inside llama-ui, stored only in IndexedDB.
 *   - `user`    — filesystem-sourced from the user's global skill
 *                 directory (~/.pi/agent/skills/ in Pi's layout).
 *   - `project` — filesystem-sourced from a project-local directory
 *                 (./.pi/skills/ or ./.agents/skills/).
 *
 * Mirrors Pi's `SourceInfo.scope` (`user` | `project`) plus a `lib`
 * stand-in for in-app authored skills that have no filesystem twin.
 * `path` (when present) corresponds to Pi's `FileInfo.path`.
 */
export type SkillOrigin = 'lib' | 'user' | 'project';

/**
 * Pluggable skill source.
 *
 * The current SkillsStore only consumes the IndexedDB-backed `lib`
 * source, but the same shape lets a future filesystem-backed loader
 * (mirroring Pi's `~/.pi/agent/skills/`, `.pi/skills/`, package
 * `skills/`) feed rows through the same pipe. Providers are read-only
 * from the UI's perspective — writes go through the in-app author or
 * the SKILL.md export pipeline.
 */
export interface SkillProvider {
	readonly source: SkillOrigin;
	list(): Promise<DatabaseSkill[]>;
	read(path: string): Promise<DatabaseSkill | null>;
	/**
	 * Optional filesystem watch. Provider implementations that can
	 * observe external changes (Pi's watcher in the agent package) will
	 * surface them here; the in-app IndexedDB provider has nothing to
	 * emit and may omit this method.
	 */
	watch?(onChange: (skills: DatabaseSkill[]) => void): () => void;
}

/**
 * Stored user-defined skill.
 *
 * A skill is the single abstraction behind what used to be called a *prompt*.
 * The same row is used as an on-demand insertable message template
 * (the chat composer "Insert skill" menu) and as an auto-injected system
 * message when marked always-on (read from `skillPreferencesStore` at
 * chat start — see `composeSystemPromptWithAlwaysOnSkills` in
 * `chat.svelte.ts`).
 *
 * UI preferences (currently just `alwaysOn`) live separately in
 * `skillPreferencesStore` (localStorage) so this row stays a faithful
 * Agent Skills payload: export round-trips through `SKILL.md` without
 * llama-ui configuration leaking into the file.
 *
 * Field names deliberately mirror the Agent Skills standard
 * (https://agentskills.io/specification) so a row can round-trip through
 * a SKILL.md export and interoperate with Pi's skill loader out of the
 * box.
 *
 * Required fields (Agent Skills spec):
 *   - `name` — human-facing identifier, normalized to lowercase + hyphens
 *   - `description` — short summary (required by Pi for auto-load)
 *   - `content` — the instruction body
 *
 * Optional fields (Agent Skills spec):
 *   - `license` — license name or reference
 *   - `compatibility` — environment requirements (max 500 chars)
 *   - `metadata` — arbitrary key-value mapping (stored as JSON string)
 *   - `allowedTools` — space-delimited list of pre-approved tools
 *   - `disableModelInvocation` — when true, hidden from system prompt
 *
 * Internal fields (not in spec, used by llama-ui):
 *   - `id` — row primary key. UUID for in-app authored rows; derived
 *            from `path` for filesystem-sourced rows so identity is
 *            stable across reloads.
 *   - `path` — absolute filesystem path of the SKILL.md (filesystem-
 *              sourced rows only).
 *   - `origin` — provenance tag: `lib` for in-app authored rows,
 *                `user` / `project` for filesystem-sourced rows.
 *   - `lastModified` — ISO timestamp for sort order
 */
/**
 * In-app row alias for `DatabaseSkill` — exported as a named type so
 * consumers (chat store, chat message components) can refer to the
 * library row shape without re-importing the store module.
 */
export type Skill = DatabaseSkill;

export interface DatabaseSkill {
	id: string;
	name: string;
	/** Short summary required by both the llama-ui store and Pi auto-load. */
	description: string;
	content: string;
	lastModified: number;
	path?: string;
	origin: SkillOrigin;
	license?: string;
	compatibility?: string;
	metadata?: Record<string, string> | string;
	allowedTools?: string;
	disableModelInvocation?: boolean;
}

export interface McpServerOverride {
	serverId: string;
	enabled?: boolean;
}

export interface DatabaseConversation {
	id: string;
	name: string;
	lastModified: number;
	currNode: string;
	forkedFromConversationId?: string;
	pinned?: boolean;
	mcpServerOverrides?: McpServerOverride[];
}

export interface DatabaseMessageExtraAudioFile {
	type: AttachmentType.AUDIO;
	name: string;
	size?: number;
	base64Data: string;
	mimeType: string;
}

export interface DatabaseMessageExtraVideoFile {
	type: AttachmentType.VIDEO;
	name: string;
	size?: number;
	base64Data: string;
	mimeType: string;
}

export interface DatabaseMessageExtraImageFile {
	type: AttachmentType.IMAGE;
	name: string;
	size?: number;
	base64Url: string;
}

/**
 * Legacy format from the old UI — pasted content was stored as "context" type
 * @deprecated Use DatabaseMessageExtraTextFile instead
 */
export interface DatabaseMessageExtraLegacyContext {
	type: AttachmentType.LEGACY_CONTEXT;
	name: string;
	size?: number;
	content: string;
}

export interface DatabaseMessageExtraPdfFile {
	type: AttachmentType.PDF;
	base64Data: string;
	name: string;
	size?: number;
	content: string;
	images?: string[];
	processedAsImages: boolean;
}

export interface DatabaseMessageExtraTextFile {
	type: AttachmentType.TEXT;
	name: string;
	size?: number;
	content: string;
}

export interface DatabaseMessageExtraMcpPrompt {
	type: AttachmentType.MCP_PROMPT;
	name: string;
	size?: number;
	serverName: string;
	promptName: string;
	content: string;
	arguments?: Record<string, string>;
}

export interface DatabaseMessageExtraMcpResource {
	type: AttachmentType.MCP_RESOURCE;
	name: string;
	size?: number;
	uri: string;
	serverName: string;
	content: string;
	mimeType?: string;
}

/**
 * Snapshot of skill metadata carried on a system message.
 *
 * The system message body holds only the skill's `content` field
 * (pure text, no scaffold). Everything else — name, description,
 * origin, path — is duplicated here as a metadata snapshot so the
 * chat UI can render the skill card without re-resolving the row
 * from the library at every render. If the user later edits the
 * library, the snapshot can drift; the parent derives `skillIsStale`
 * and surfaces a Sync affordance.
 */
export interface DatabaseMessageExtraSkill {
	type: AttachmentType.SKILL;
	skillId: string;
	title: string;
	description?: string;
	origin?: SkillOrigin;
	path?: string;
}

export type DatabaseMessageExtra =
	| DatabaseMessageExtraImageFile
	| DatabaseMessageExtraTextFile
	| DatabaseMessageExtraAudioFile
	| DatabaseMessageExtraVideoFile
	| DatabaseMessageExtraPdfFile
	| DatabaseMessageExtraMcpPrompt
	| DatabaseMessageExtraMcpResource
	| DatabaseMessageExtraSkill
	| DatabaseMessageExtraLegacyContext;

export interface DatabaseMessage {
	id: string;
	convId: string;
	type: ChatMessageType;
	timestamp: number;
	role: ChatRole;
	content: string;
	parent: string | null;
	/**
	 * @deprecated - left for backward compatibility
	 */
	thinking?: string;
	/** Reasoning content produced by the model (separate from visible content) */
	reasoningContent?: string;
	/** Serialized JSON array of tool calls made by assistant messages */
	toolCalls?: string;
	/** Chat completion id streamed by the server, used to target realtime control (e.g. end reasoning) */
	completionId?: string;
	/** Tool call ID for tool result messages (role: 'tool') */
	toolCallId?: string;
	children: string[];
	extra?: DatabaseMessageExtra[];
	timings?: ChatMessageTimings;
	model?: string;
}

export type ExportedConversation = {
	conv: DatabaseConversation;
	messages: DatabaseMessage[];
};

export type ExportedConversations = ExportedConversation | ExportedConversation[];
