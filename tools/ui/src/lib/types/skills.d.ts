/**
 * Agent Skills API types (llama-server `--skills`).
 *
 * The server owns discovery, resolution, identity, containment, parsing, and
 * XML serialization. These types carry no host paths, roots, or
 * client-constructible identities; `id` is an opaque comparison key.
 */

/** Safe, path-free diagnostic emitted by the Skills API. */
export interface SkillDiagnostic {
	severity: 'warning' | 'error';
	code: string;
	name?: string;
	scope?: string;
	provider?: string;
	message: string;
}

/** Instruction measurement facts for one catalog entry. */
export interface SkillInstructionFacts {
	bytes: number;
	lines: number;
	tokens: number;
	tokens_estimated: boolean;
	modified_at: string | null;
}

/** Bounded resource summary for one catalog entry. */
export interface SkillResourceSummary {
	count: number;
	truncated: boolean;
}

/** One entry of the deterministic GET /skills catalog. */
export interface SkillCatalogEntry {
	id: string;
	name: string;
	description: string;
	scope: 'global' | 'project';
	provider: string;
	instruction: SkillInstructionFacts;
	resources: SkillResourceSummary;
	catalog_xml: string;
}

/** GET /skills response body. */
export interface SkillCatalogResponse {
	skills: SkillCatalogEntry[];
	catalog_instruction_xml: string;
	diagnostics: SkillDiagnostic[];
}

/** Shared handler error envelope returned by Skills routes. */
export interface SkillErrorResponse {
	error: {
		code: number;
		message: string;
		type: string;
	};
}

/** POST /skills/read request body: only name and an optional safe relative path. */
export interface SkillReadRequest {
	name: string;
	path?: string;
}

/** Opaque server-owned skill identity; never decoded or constructed client-side. */
export interface SkillIdentity {
	id: string;
	name: string;
	scope: 'global' | 'project';
	provider: string;
}

/**
 * Safe, server-returned facts shown during a Skills consent pause: the
 * resolved identity plus the requested action. Never carries host paths.
 */
export interface SkillConsentInfo {
	name: string;
	scope: 'global' | 'project';
	provider: string;
	/** Requested resource path for `read_skill(name, path)`; absent for base reads. */
	path?: string;
}

/** Structured skill metadata for the full supported field set. */
export interface SkillMetadata {
	name?: string;
	description?: string;
	license?: string;
	compatibility?: string;
	metadata?: Record<string, string>;
	allowed_tools?: string;
}

/** POST /skills/read base SKILL.md result. */
export interface SkillBaseReadResult {
	kind: 'skill';
	skill: SkillIdentity & { metadata?: SkillMetadata };
	resources: { paths: string[]; truncated: boolean };
	source: string;
	body_markdown: string;
	content_xml: string;
	diagnostics: SkillDiagnostic[];
}

/** POST /skills/read resource result. */
export interface SkillResourceReadResult {
	kind: 'resource';
	skill: SkillIdentity;
	resource: { path: string };
	content_xml: string;
	diagnostics: SkillDiagnostic[];
}

/** Union of the two POST /skills/read result kinds. */
export type SkillReadResult = SkillBaseReadResult | SkillResourceReadResult;

/** Immutable per-run snapshot of one successful catalog response. */
export interface SkillRunSnapshot {
	/** Selected CWD the snapshot was resolved under; undefined = server process CWD. */
	cwd: string | undefined;
	/** The issuing run's own frozen catalog response. */
	catalog: SkillCatalogResponse;
	/** Immutable entry copies in server order. */
	entries: readonly SkillCatalogEntry[];
	/** Serialized complete `<skills_catalog total="..." included="...">` envelope (pre-budget). */
	envelope: string;
	/** Entry count of the frozen snapshot (server order). */
	total: number;
}

/** How catalog packing measures token usage. */
export type SkillPackingMode = 'direct' | 'estimated';

/** Inputs to budgeted catalog packing. */
export interface SkillPackOptions {
	/** maxSkillBudget: non-negative integer; a literal zero disables the envelope. */
	budget: number;
	/**
	 * 'direct' measures with the selected-model tokenizer request; 'estimated'
	 * uses the deterministic estimate policy and never issues tokenizer
	 * requests. The selected model is used only in direct mode.
	 */
	mode: SkillPackingMode;
	/** Selected model id; required for direct-mode measurement. */
	model?: string;
	signal?: AbortSignal;
}

/** Result of applying maxSkillBudget to a frozen snapshot. */
export interface SkillPackedCatalog {
	/** Budgeted `<skills_catalog>` envelope; '' when zero budget or empty catalog. */
	envelope: string;
	/** Total entries in the frozen snapshot (server order). */
	total: number;
	/** Entries retained after budgeting; included < total marks a partial envelope. */
	included: number;
	/** True when the measurement is a labeled deterministic estimate. */
	estimated: boolean;
}
