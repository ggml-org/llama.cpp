// Shared Skills fixtures for the unit and client suites.

import type {
	SkillBaseReadResult,
	SkillCatalogEntry,
	SkillCatalogResponse,
	SkillReadResult,
	SkillResourceReadResult
} from '$lib/types';

export function jsonResponse(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		headers: { 'content-type': 'application/json' },
		status
	});
}

export function makeEntry(
	name: string,
	overrides: Partial<SkillCatalogEntry> = {}
): SkillCatalogEntry {
	return {
		catalog_xml: `<skill><name>${name}</name></skill>`,
		description: `description of ${name}`,
		id: `opaque-${name}`,
		instruction: { bytes: 16, lines: 1, modified_at: null, tokens: 4, tokens_estimated: true },
		name,
		provider: 'agents',
		resources: { count: 0, truncated: false },
		scope: 'project',
		...overrides
	};
}

export function makeCatalog(
	entries: SkillCatalogEntry[],
	instructionXml = '<available_skills>Call read_skill(name) when matching.</available_skills>'
): SkillCatalogResponse {
	return { catalog_instruction_xml: instructionXml, diagnostics: [], skills: entries };
}

export function catalogOf(...names: string[]): SkillCatalogResponse {
	return makeCatalog(names.map((name) => makeEntry(name)));
}

export function baseResult(
	name = 'example-skill',
	overrides: Partial<SkillBaseReadResult> = {}
): SkillBaseReadResult {
	return {
		body_markdown: `# Body of ${name}`,
		content_xml: `<skill_content name="${name}">body</skill_content>`,
		diagnostics: [],
		kind: 'skill',
		resources: { paths: [], truncated: false },
		skill: {
			id: `opaque-${name}`,
			metadata: { description: `description of ${name}` },
			name,
			provider: 'agents',
			scope: 'project'
		},
		source: `---\nname: ${name}\n---\n# Body of ${name}`,
		...overrides
	};
}

export function resourceResult(
	name = 'example-skill',
	path = 'references/DETAILS.md',
	overrides: Partial<SkillResourceReadResult> = {}
): SkillResourceReadResult {
	return {
		content_xml: `<skill_resource name="${name}" path="${path}">data</skill_resource>`,
		diagnostics: [],
		kind: 'resource',
		resource: { path },
		skill: { id: `opaque-${name}`, name, provider: 'agents', scope: 'project' },
		source: 'data',
		...overrides
	};
}

/** Preview-shaped read result carrying distinct markdown body and raw source. */
export function previewResult(name: string): SkillReadResult {
	return baseResult(name, {
		body_markdown: `# Content of ${name}\n\nBody text.\n`,
		source: `---\nname: ${name}\ndescription: raw frontmatter\n---\n# Content of ${name}\n\nBody text.\n`
	});
}

/** Controllable fetch double that rejects the in-flight request on abort. */
export function deferredRead() {
	const state: { signal?: AbortSignal } = {};
	const { promise, reject, resolve } = Promise.withResolvers<Response>();

	return {
		attach(init?: RequestInit) {
			state.signal = init?.signal ?? undefined;
			init?.signal?.addEventListener(
				'abort',
				() => reject(new DOMException('The operation was aborted.', 'AbortError')),
				{ once: true }
			);
		},
		promise,
		reject,
		resolve,
		get signal() {
			return state.signal;
		}
	};
}
