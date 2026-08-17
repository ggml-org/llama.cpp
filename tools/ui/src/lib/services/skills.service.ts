import { API_SKILLS, API_TOKENIZE, X_SKILL_CWD_HEADER } from '$lib/constants';
import type {
	SkillCatalogEntry,
	SkillCatalogResponse,
	SkillPackOptions,
	SkillPackedCatalog,
	SkillReadRequest,
	SkillReadResult,
	SkillRunSnapshot
} from '$lib/types';
import { apiFetch } from '$lib/utils';

/**
 * SkillsService - stateless transport and catalog packing for the dedicated
 * llama-server Skills API.
 *
 * The server owns host-filesystem discovery, resolution, identity,
 * containment, parsing, and XML. This service sends only the supported
 * name/path inputs, keeps CWD handling consistent, and applies the shared
 * catalog snapshot and budget policy.
 */
export class SkillsService {
	/**
	 * Fetch the deterministic catalog for an effective CWD.
	 *
	 * An absent `cwd` means the canonical server process CWD; a selected
	 * non-whitespace CWD is sent as `X-Skill-Cwd`. Whitespace-only values are
	 * treated as absent before transport.
	 */
	static async list(cwd?: string, signal?: AbortSignal): Promise<SkillCatalogResponse> {
		return apiFetch<SkillCatalogResponse>(API_SKILLS.LIST, {
			headers: skillCwdHeaders(cwd),
			signal
		});
	}

	/**
	 * Read a skill's current base content or a resource.
	 *
	 * Sends exactly `{ name }` or `{ name, path }` — never identity, scope,
	 * provider, root, or absolute path. The server re-resolves the name for the
	 * effective CWD on every request.
	 */
	static async read(
		request: SkillReadRequest,
		cwd?: string,
		signal?: AbortSignal
	): Promise<SkillReadResult> {
		return apiFetch<SkillReadResult>(API_SKILLS.READ, {
			body: JSON.stringify(
				request.path !== undefined
					? { name: request.name, path: request.path }
					: { name: request.name }
			),
			headers: skillCwdHeaders(cwd),
			method: 'POST',
			signal
		});
	}
}

/**
 * Map a selected CWD to the Skills request header, or absent when no
 * non-whitespace CWD is selected (the server then uses the process CWD).
 */
interface SkillTokenizeResponse {
	tokens: number[];
}

const SKILL_CATALOG_TAG = 'skills_catalog';
const SKILL_CATALOG_TOTAL_ATTR = 'total';
const SKILL_CATALOG_INCLUDED_ATTR = 'included';

/**
 * Documented deterministic estimate policy shared with the server: the integer
 * ceiling of UTF-8 bytes / 4. Used in estimated mode and as the labeled
 * fallback when the direct tokenizer is unavailable.
 */
export function estimateSkillTokens(text: string): number {
	return Math.ceil(new TextEncoder().encode(text).length / 4);
}

/**
 * Shared direct/estimated pack-option resolution used by agentic runs and the
 * catalog presentation, so both present the same budget policy. Direct mode
 * requires a non-empty effective model and, in router mode, a loaded model.
 */
export function resolveSkillPackOptions(
	effectiveModel: string,
	routerMode: boolean,
	isModelLoaded: (model: string) => boolean
): Pick<SkillPackOptions, 'mode' | 'model'> {
	const directOk = effectiveModel !== '' && (!routerMode || isModelLoaded(effectiveModel));

	return directOk ? { mode: 'direct', model: effectiveModel } : { mode: 'estimated' };
}

/**
 * Serialize the complete `<skills_catalog total="..." included="...">`
 * envelope: the server's `catalog_instruction_xml` fragment verbatim first,
 * then each entry's `catalog_xml` fragment in server order, never re-escaped.
 */
export function serializeSkillCatalogEnvelope(catalog: SkillCatalogResponse): string {
	const { catalog_instruction_xml, skills } = catalog;

	return serializeEnvelope(catalog_instruction_xml, skills, skills.length, skills.length);
}

/**
 * Build an immutable per-run snapshot from a run's own successful catalog
 * response. Entries are deep-copied and frozen, so no later store refresh or
 * CWD change can reach the snapshot.
 */
export function buildSkillRunSnapshot(
	cwd: string | undefined,
	catalog: SkillCatalogResponse,
	disabledIds?: ReadonlySet<string>
): SkillRunSnapshot {
	// The model-facing view excludes manual-only skills and any locally
	// disabled opaque IDs from the entries, the envelope, and the budget
	// count; the raw catalog stays available for the UI listing and the
	// explicit /skills picker.
	const modelEntries = catalog.skills.filter(
		(entry) => !entry.disable_model_invocation && !disabledIds?.has(entry.id)
	);
	const entries = Object.freeze(modelEntries.map(freezeEntry));

	return {
		catalog,
		cwd,
		entries,
		envelope: serializeEnvelope(
			catalog.catalog_instruction_xml,
			entries,
			entries.length,
			entries.length
		),
		total: entries.length
	};
}

function freezeEntry(entry: SkillCatalogEntry): SkillCatalogEntry {
	return Object.freeze({
		...entry,
		instruction: Object.freeze({ ...entry.instruction }),
		resources: Object.freeze({ ...entry.resources })
	});
}

/**
 * SkillsPackingService - centralized budget packing policy for the frozen
 * catalog envelope.
 *
 * Direct mode measures with the audited selected-model tokenizer request and
 * no-special-token flags; a failed or unavailable request falls back to the
 * labeled deterministic estimate without retry, model selection, or wake.
 * Estimated mode never issues a tokenizer request.
 */
export class SkillsPackingService {
	/**
	 * Exact token count for `content` via POST /tokenize with the selected
	 * model, using the audited no-special-token flags (the /tokenize defaults).
	 */
	static async countTokens(content: string, model: string, signal?: AbortSignal): Promise<number> {
		const response = await apiFetch<SkillTokenizeResponse>(API_TOKENIZE, {
			body: JSON.stringify({ add_special: false, content, model, parse_special: true }),
			method: 'POST',
			signal
		});

		return response.tokens.length;
	}

	/**
	 * Apply maxSkillBudget to a frozen snapshot envelope. The complete envelope
	 * is measured first; entry fragments are then truncated at the budget
	 * boundary in server order, always keeping the instruction fragment. A
	 * literal zero budget or an empty catalog produces no envelope.
	 */
	static async pack(
		snapshot: SkillRunSnapshot,
		options: SkillPackOptions
	): Promise<SkillPackedCatalog> {
		const { budget, mode, model, signal } = options;
		const total = snapshot.total;

		if (budget <= 0 || total === 0) {
			return { envelope: '', estimated: false, fullTokens: null, included: 0, total };
		}

		if (mode === 'direct' && model) {
			try {
				return await packWithMeasure(
					snapshot,
					(text) => SkillsPackingService.countTokens(text, model, signal),
					budget,
					false
				);
			} catch {
				// Tokenizer request failed or was unavailable at run time: fall
				// back to the labeled deterministic estimate. Never retry, never
				// select or wake a model.
				return packWithMeasure(snapshot, estimateSkillTokens, budget, true);
			}
		}

		// Estimated mode (or direct mode without a selected model): the
		// documented deterministic estimate policy, distinctly labeled.
		return packWithMeasure(snapshot, estimateSkillTokens, budget, true);
	}
}

/**
 * Measure the complete envelope, then walk leading entry fragments in server
 * order to find the budget boundary. `measure` is exact (direct mode) or the
 * deterministic estimate; both grow monotonically with the serialized prefix.
 * The complete envelope is measured exactly once: the final prefix equals the
 * complete envelope and is known to exceed the budget here, so the walk stops
 * before it.
 */
async function packWithMeasure(
	snapshot: SkillRunSnapshot,
	measure: (text: string) => number | Promise<number>,
	budget: number,
	estimated: boolean
): Promise<SkillPackedCatalog> {
	const { catalog, entries, envelope, total } = snapshot;
	const fullTokens = await measure(envelope);

	if (fullTokens <= budget) {
		return { envelope, estimated, fullTokens, included: total, total };
	}

	const instructionXml = catalog.catalog_instruction_xml;

	let included = 0;

	for (let i = 1; i < total; i++) {
		if ((await measure(serializeEnvelope(instructionXml, entries, total, i))) > budget) {
			break;
		}

		included = i;
	}

	return {
		envelope: serializeEnvelope(instructionXml, entries, total, included),
		estimated,
		fullTokens,
		included,
		total
	};
}

function serializeEnvelope(
	instructionXml: string,
	entries: readonly SkillCatalogEntry[],
	total: number,
	included: number
): string {
	return `<${SKILL_CATALOG_TAG} ${SKILL_CATALOG_TOTAL_ATTR}="${total}" ${SKILL_CATALOG_INCLUDED_ATTR}="${included}">${instructionXml}${entries
		.slice(0, included)
		.map((entry) => entry.catalog_xml)
		.join('')}</${SKILL_CATALOG_TAG}>`;
}
function skillCwdHeaders(cwd: string | undefined): Record<string, string> | undefined {
	if (cwd === undefined || cwd.trim().length === 0) {
		return undefined;
	}

	return { [X_SKILL_CWD_HEADER]: cwd };
}
