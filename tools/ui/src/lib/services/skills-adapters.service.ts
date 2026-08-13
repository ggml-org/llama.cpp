import { JsonSchemaType, MessageRole, ToolCallType, ToolPermissionDecision } from '$lib/enums';
import { SkillsService } from '$lib/services/skills.service';
import type { AgenticToolCallPayload } from '$lib/types/agentic';
import type { ApiChatMessageData } from '$lib/types/api';
import type {
	OpenAIToolDefinition,
	ToolExecutionResult
} from '$lib/types/mcp';
import type { DatabaseMessage, DatabaseMessageExtra, DatabaseMessageExtraSkill } from '$lib/types/database';
import type {
	SkillCatalogEntry,
	SkillConsentInfo,
	SkillPackedCatalog,
	SkillReadResult,
	SkillRunSnapshot
} from '$lib/types/skills';

/** Model-facing tool names owned by the Skills adapters. */
export const SKILL_LIST_TOOL = 'list_skill';
export const SKILL_READ_TOOL = 'read_skill';
/** Display label for the Skills tool source in the established consent UI. */
export const SKILL_SERVER_LABEL = 'Skills';

const SKILL_ADAPTER_COLLISION_CODE = 'skill_adapter_collision';

/** Safe diagnostic emitted when a Skills adapter name collides with an existing tool. */
export interface SkillAdapterDiagnostic {
	code: string;
	name?: string;
	message: string;
}

/** Result of snapshot-authorized adapter registration. */
export interface SkillAdaptersBuildResult {
	definitions: OpenAIToolDefinition[];
	diagnostics: SkillAdapterDiagnostic[];
}

/**
 * Input to the shared successful-base-activation operation. Both the
 * model consent path (approved `read_skill` base reads) and the explicit
 * `/skills <name>` path route through it; the durable record is keyed by
 * conversation plus the exact opaque server identity.
 */
export interface SkillActivationInput {
	conversationId: string;
	/** Successful server read; only `kind: 'skill'` results persist an activation. */
	result: SkillReadResult;
	/** CWD the read resolved under; used only for in-flight metadata, never persisted. */
	cwd?: string;
	/**
	 * Model path: the model's own tool call id. The store anchors the paired
	 * tool result to the persisted assistant message carrying this call id.
	 * Absent on the slash path, which creates a synthetic assistant pair.
	 */
	toolCallId?: string;
}

/** Outcome of the shared activation operation. */
export interface SkillActivationResult {
	/** Typed durable SKILL metadata for the persisted tool result message. */
	extra: DatabaseMessageExtraSkill;
	/** True when this call created a NEW durable record; false on dedupe or session-only resource approval. */
	created: boolean;
	/**
	 * The persisted tool result message when the operation created it (slash
	 * path, and model path via the store-anchored pair); null otherwise, in
	 * which case the caller persists the message with `extra` attached.
	 */
	toolResultMessage: DatabaseMessage | null;
}

/**
 * Durable successful-base-activation boundary.
 *
 * Task 4 replaces the Task 3 in-memory per-run seam with the shared
 * successful-base persistence operation: activations are reconstructed from
 * the conversation's persisted typed SKILL metadata, keyed by the exact
 * opaque server identity, so an approval survives runs and reloads. Only
 * successful base reads persist; denial, failure, and unavailability record
 * nothing, and resource approvals are session-scoped.
 */
export interface SkillActivationStore {
	/** True when the conversation holds a durable base activation for the exact opaque id. */
	isActivated(conversationId: string, identityId: string): boolean;
	/** Load the conversation's persisted activations into the store's cache (e.g. at run start). */
	loadConversation(conversationId: string): Promise<void>;
	/** Record a successful base activation through the single shared persistence path. */
	recordActivation(input: SkillActivationInput): Promise<SkillActivationResult>;
}

/** Inputs for one run's Skills adapters. */
export interface SkillRunAdaptersOptions {
	snapshot: SkillRunSnapshot;
	packed: SkillPackedCatalog;
	definitions: OpenAIToolDefinition[];
	/** Conversation the run belongs to; the durable activation store is keyed by it. */
	conversationId: string;
	/** The durable conversation-scoped activation store (Task 4). */
	activation: SkillActivationStore;
	/**
	 * The established consent mechanism: pauses for an explicit allow/deny
	 * decision and resolves it. `skill` carries only safe server-returned
	 * identity facts for the consent card.
	 */
	requestPermission: (
		toolName: string,
		serverLabel: string,
		skill?: SkillConsentInfo,
		signal?: AbortSignal
	) => Promise<ToolPermissionDecision>;
}

/**
 * Result of one `SkillRunAdapters.execute` call. Adds the durable-activation
 * hand-off to the generic tool result: when the shared operation created the
 * paired tool result message, `activationRecorded` + `recordedToolResultMessageId`
 * tell the agentic flow to reuse it instead of creating a second message;
 * `extras` carry the typed SKILL metadata for messages the flow persists.
 */
export interface SkillToolExecutionResult extends ToolExecutionResult {
	/** True when a NEW durable base activation was persisted by the shared store. */
	activationRecorded?: boolean;
	/** The tool result message the store created for the activation. */
	recordedToolResultMessageId?: string;
	/** Typed SKILL metadata to attach to the flow-persisted tool result message. */
	extras?: DatabaseMessageExtra[];
}

/**
 * Register snapshot-authorized Skills adapters.
 *
 * A zero-budget or empty envelope registers nothing. A complete envelope
 * exposes only `read_skill`; a partial envelope also exposes `list_skill`.
 * `read_skill` carries a dynamic `enum` of frozen snapshot names; existing
 * non-Skills/custom/MCP names win collisions and the colliding adapter is
 * omitted with a safe diagnostic.
 */
export function buildSkillToolDefinitions(
	snapshot: SkillRunSnapshot,
	packed: SkillPackedCatalog,
	existingNames: ReadonlySet<string>
): SkillAdaptersBuildResult {
	if (packed.envelope === '') {
		return { definitions: [], diagnostics: [] };
	}

	const names = snapshot.entries.map((entry) => entry.name);
	const diagnostics: SkillAdapterDiagnostic[] = [];
	const definitions: OpenAIToolDefinition[] = [];

	const register = (def: OpenAIToolDefinition) => {
		if (existingNames.has(def.function.name)) {
			diagnostics.push({
				code: SKILL_ADAPTER_COLLISION_CODE,
				message: `Skills tool "${def.function.name}" collides with an existing tool and was not registered.`,
				name: def.function.name
			});

			return;
		}

		definitions.push(freezeDefinition(def));
	};

	register(readSkillDefinition(names));

	if (packed.included < packed.total) {
		register(listSkillDefinition());
	}

	return { definitions, diagnostics };
}

/**
 * Decorate a run's first-request messages with the budgeted `<skills_catalog>`
 * envelope, byte-preserved. The envelope is appended to the first system
 * message, or prepended as a system message when the run has none. The input
 * array is never mutated and an empty envelope leaves it untouched.
 */
export function decorateSkillPrompt(
	messages: ApiChatMessageData[],
	envelope: string
): ApiChatMessageData[] {
	if (envelope === '') return messages;

	const decorated = messages.map((message) => ({ ...message }));
	const system = decorated.find((message) => message.role === MessageRole.SYSTEM);

	if (system && typeof system.content === 'string') {
		system.content = system.content ? `${system.content}\n${envelope}` : envelope;

		return decorated;
	}

	return [{ content: envelope, role: MessageRole.SYSTEM }, ...decorated];
}

/**
 * Structured `list_skill()` result: snapshot entries only (`name`,
 * `description`, `scope`, `provider`), never XML or identities.
 */
export function listSkillContent(entries: readonly SkillCatalogEntry[]): string {
	return JSON.stringify(
		entries.map((entry) => ({
			name: entry.name,
			description: entry.description,
			scope: entry.scope,
			provider: entry.provider
		}))
	);
}

/** Structured no-content/no-activation denial result (never carries XML). */
export function skillDenialResult(toolName: string): string {
	return JSON.stringify({
		message: 'Skill access was denied by the user.',
		status: 'denied',
		tool: toolName
	});
}

/** Structured error result (never carries XML). */
export function skillErrorResult(toolName: string, message: string): string {
	return JSON.stringify({ message, status: 'error', tool: toolName });
}

/**
 * One run's Skills adapters. Exists only when the run's frozen snapshot
 * authorized at least one adapter; it holds no mutable catalog state, never
 * refreshes the catalog, and derives identity only from server responses.
 */
export class SkillRunAdapters {
	private readonly _snapshot: SkillRunSnapshot;
	private readonly _packed: SkillPackedCatalog;
	private readonly _definitions: readonly OpenAIToolDefinition[];
	private readonly _snapshotNames: Set<string>;
	private readonly _registeredNames: Set<string>;
	private readonly _conversationId: string;
	private readonly _activation: SkillActivationStore;
	private readonly _requestPermission: SkillRunAdaptersOptions['requestPermission'];
	/** One shared pending consent decision per consent identity. */
	private readonly _pendingDecisions = new Map<string, Promise<'allowed' | 'denied'>>();

	constructor(options: SkillRunAdaptersOptions) {
		this._snapshot = options.snapshot;
		this._packed = options.packed;
		this._definitions = options.definitions;
		this._snapshotNames = new Set(options.snapshot.entries.map((entry) => entry.name));
		this._registeredNames = new Set(options.definitions.map((def) => def.function.name));
		this._conversationId = options.conversationId;
		this._activation = options.activation;
		this._requestPermission = options.requestPermission;
	}

	get definitions(): readonly OpenAIToolDefinition[] {
		return this._definitions;
	}

	get envelope(): string {
		return this._packed.envelope;
	}

	/** True only for skill tool names this run actually registered (collision-free). */
	isSkillTool(name: string): boolean {
		return this._registeredNames.has(name);
	}

	decorate(messages: ApiChatMessageData[]): ApiChatMessageData[] {
		return decorateSkillPrompt(messages, this.envelope);
	}

	async execute(
		toolCall: AgenticToolCallPayload,
		signal?: AbortSignal
	): Promise<SkillToolExecutionResult> {
		const name = toolCall.function.name;
		let args: Record<string, unknown>;

		try {
			args = parseSkillArguments(toolCall.function.arguments);
		} catch {
			return {
				content: skillErrorResult(name, `Malformed tool call arguments for "${name}".`),
				isError: true
			};
		}

		if (name === SKILL_LIST_TOOL) {
			return { content: listSkillContent(this._snapshot.entries), isError: false };
		}

		if (name === SKILL_READ_TOOL) {
			return this.executeRead(args, signal, toolCall.id);
		}

		return {
			content: skillErrorResult(name, `Unknown Skills tool "${name}".`),
			isError: true
		};
	}

	/**
	 * Resolve a model read through the server using only the snapshot
	 * CWD/name/path, then gate the resolved identity through consent.
	 * Names outside the frozen snapshot never dispatch a request.
	 */
	private async executeRead(
		args: Record<string, unknown>,
		signal?: AbortSignal,
		toolCallId?: string
	): Promise<SkillToolExecutionResult> {
		const name = args.name;
		const path = args.path;

		if (typeof name !== 'string' || name.length === 0) {
			return {
				content: skillErrorResult(SKILL_READ_TOOL, 'read_skill requires a string "name" argument.'),
				isError: true
			};
		}

		if (path !== undefined && (typeof path !== 'string' || path.length === 0)) {
			return {
				content: skillErrorResult(SKILL_READ_TOOL, 'read_skill "path" must be a non-empty string when provided.'),
				isError: true
			};
		}

		if (!this._snapshotNames.has(name)) {
			return {
				content: skillErrorResult(SKILL_READ_TOOL, `Unknown skill "${name}" in this run's catalog snapshot.`),
				isError: true
			};
		}

		let result: SkillReadResult;

		try {
			result = await SkillsService.read({ name, path }, this._snapshot.cwd, signal);
		} catch (error) {
			// A failed read is never consent and persists no content or metadata.
			const message = error instanceof Error ? error.message : String(error);

			return {
				content: skillErrorResult(SKILL_READ_TOOL, `Skills read failed: ${message}`),
				isError: true
			};
		}

		const decision = await this.authorize(result, path, signal);

		if (decision === 'denied') {
			return { content: skillDenialResult(SKILL_READ_TOOL), isError: true };
		}

		// Allowed: route through the single shared durable persistence
		// operation. A NEW base activation persists the paired tool result
		// message itself; deduped base reads and session-only resource
		// approvals hand the typed metadata back for the flow to attach.
		const record = await this._activation.recordActivation({
			conversationId: this._conversationId,
			result,
			cwd: this._snapshot.cwd,
			...(toolCallId !== undefined ? { toolCallId } : {})
		});

		if (record.created && record.toolResultMessage) {
			return {
				content: result.content_xml,
				isError: false,
				activationRecorded: true,
				recordedToolResultMessageId: record.toolResultMessage.id,
				extras: [record.extra]
			};
		}

		// Server XML is opaque model content, preserved byte-for-byte.
		return { content: result.content_xml, isError: false, extras: [record.extra] };
	}

	/**
	 * Per-identity consent: an identity with a durable activation proceeds;
	 * an unapproved one pauses in the established consent mechanism.
	 * Concurrent reads of the same resolved identity share one pending
	 * decision; only explicit allow resumes past the pause.
	 */
	private async authorize(
		result: SkillReadResult,
		path: string | undefined,
		signal?: AbortSignal
	): Promise<'allowed' | 'denied'> {
		const identityId = result.skill.id;

		if (this._activation.isActivated(this._conversationId, identityId)) return 'allowed';

		const pending = this._pendingDecisions.get(identityId);

		if (pending) return pending;

		const decision = this._requestPermission(
			SKILL_READ_TOOL,
			SKILL_SERVER_LABEL,
			{
				name: result.skill.name,
				scope: result.skill.scope,
				provider: result.skill.provider,
				...(path !== undefined ? { path } : {})
			},
			signal
		)
			.then(async (permission) => {
				// Yield so Svelte can flush the consent card before the result lands.
				await new Promise((resolve) => setTimeout(resolve, 0));

				if (signal?.aborted || permission === ToolPermissionDecision.DENY) {
					return 'denied' as const;
				}

				return 'allowed' as const;
			})
			.finally(() => {
				this._pendingDecisions.delete(identityId);
			});

		this._pendingDecisions.set(identityId, decision);

		return decision;
	}
}

function parseSkillArguments(raw: string): Record<string, unknown> {
	const trimmed = raw.trim();

	if (trimmed === '') return {};

	return JSON.parse(trimmed) as Record<string, unknown>;
}

function readSkillDefinition(names: string[]): OpenAIToolDefinition {
	return {
		type: ToolCallType.FUNCTION,
		function: {
			description:
				'Read the current base content of a skill by name, or one of its resources by a relative path.',
			name: SKILL_READ_TOOL,
			parameters: {
				type: JsonSchemaType.OBJECT,
				properties: {
					name: { type: 'string', enum: names },
					path: { type: 'string' }
				},
				required: ['name']
			}
		}
	};
}

function listSkillDefinition(): OpenAIToolDefinition {
	return {
		type: ToolCallType.FUNCTION,
		function: {
			description: 'List the skills available in this run, with their descriptions.',
			name: SKILL_LIST_TOOL,
			parameters: {
				type: JsonSchemaType.OBJECT,
				properties: {},
				required: []
			}
		}
	};
}

function freezeDefinition(def: OpenAIToolDefinition): OpenAIToolDefinition {
	return Object.freeze({
		...def,
		function: Object.freeze({
			...def.function,
			parameters: Object.freeze({ ...def.function.parameters })
		})
	});
}
