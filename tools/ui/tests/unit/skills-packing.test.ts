import {
	normalizeSkillBudget,
	POSITIVE_INTEGER_FIELDS,
	SETTING_CONFIG_DEFAULT,
	SETTINGS_CHAT_SECTIONS,
	SETTINGS_KEYS,
	SETTINGS_SECTION_SLUGS
} from '$lib/constants';
import { SettingsFieldType } from '$lib/enums/settings.enums';
import {
	buildSkillRunSnapshot,
	estimateSkillTokens,
	resolveSkillPackOptions,
	serializeSkillCatalogEnvelope,
	SkillsPackingService
} from '$lib/services/skills.service';
import type { SkillCatalogEntry, SkillCatalogResponse, SkillRunSnapshot } from '$lib/types';
import { afterEach, describe, expect, it, vi } from 'vitest';

function jsonResponse(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		headers: { 'content-type': 'application/json' },
		status
	});
}

function makeEntry(name: string, xml: string): SkillCatalogEntry {
	return {
		catalog_xml: xml,
		description: `description of ${name}`,
		id: `opaque-${name}`,
		instruction: { bytes: 16, lines: 1, modified_at: null, tokens: 4, tokens_estimated: true },
		name,
		provider: 'agents',
		resources: { count: 0, truncated: false },
		scope: 'project'
	};
}

function makeCatalog(
	entries: SkillCatalogEntry[],
	instructionXml = '<available_skills>Call read_skill(name) when a task matches.</available_skills>'
): SkillCatalogResponse {
	return { catalog_instruction_xml: instructionXml, diagnostics: [], skills: entries };
}

/** Deterministic tokenizer double: one token per character. */
function charCountingTokenizer(): ReturnType<typeof vi.fn> {
	return vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
		const body = JSON.parse(init?.body as string) as { content: string };
		const tokens = Array.from({ length: body.content.length }, (_, i) => i);

		return jsonResponse({ tokens });
	});
}

describe('estimateSkillTokens', () => {
	it('computes the integer ceiling of UTF-8 bytes / 4', () => {
		expect(estimateSkillTokens('')).toBe(0);
		expect(estimateSkillTokens('x'.repeat(10))).toBe(3);
		expect(estimateSkillTokens('x'.repeat(12))).toBe(3);
		expect(estimateSkillTokens('x'.repeat(13))).toBe(4);
	});

	it('counts UTF-8 bytes, not characters', () => {
		// U+20AC is 3 UTF-8 bytes: 12 bytes -> 3 tokens
		expect(estimateSkillTokens('\u20ac'.repeat(4))).toBe(3);
	});
});

describe('envelope serialization', () => {
	it('serializes the complete envelope with deterministic count attributes', () => {
		const catalog = makeCatalog(
			[makeEntry('one', '<one/>'), makeEntry('two', '<two/>')],
			'<inst/>'
		);

		expect(serializeSkillCatalogEnvelope(catalog)).toBe(
			'<skills_catalog total="2" included="2"><inst/><one/><two/></skills_catalog>'
		);
	});

	it('preserves server XML verbatim without re-escaping', () => {
		const raw = '<skill a="1">&amp;<b>&lt;raw&gt; &quot;text&quot;</b></skill>';
		const catalog = makeCatalog([makeEntry('escaped', raw)]);
		const envelope = serializeSkillCatalogEnvelope(catalog);

		expect(envelope).toContain(raw);
		expect(envelope).toContain('&amp;');
		expect(envelope).not.toContain('&amp;amp;');
	});
});

describe('buildSkillRunSnapshot', () => {
	it('copies entries immutably so later store mutations cannot reach the snapshot', () => {
		const catalog = makeCatalog([makeEntry('one', '<one/>'), makeEntry('two', '<two/>')]);
		const snapshot = buildSkillRunSnapshot('/w', catalog);

		catalog.skills[0].catalog_xml = 'MUTATED';
		catalog.catalog_instruction_xml = 'MUTATED';
		catalog.skills.push(makeEntry('three', '<three/>'));

		expect(snapshot.cwd).toBe('/w');
		expect(snapshot.total).toBe(2);
		expect(snapshot.entries).toHaveLength(2);
		expect(snapshot.entries[0].catalog_xml).toBe('<one/>');
		expect(snapshot.envelope).toContain('<one/>');
		expect(snapshot.envelope).not.toContain('MUTATED');
		expect(snapshot.envelope).not.toContain('<three/>');
	});

	it('freezes the entry copies and the entry array', () => {
		const snapshot = buildSkillRunSnapshot(undefined, makeCatalog([makeEntry('one', '<one/>')]));

		expect(Object.isFrozen(snapshot.entries)).toBe(true);
		expect(Object.isFrozen(snapshot.entries[0])).toBe(true);
		expect(Object.isFrozen(snapshot.entries[0].instruction)).toBe(true);
		expect(Object.isFrozen(snapshot.entries[0].resources)).toBe(true);
	});
});

describe('buildSkillRunSnapshot model-view filtering', () => {
	it('excludes manual-only entries from entries, total, and the envelope', () => {
		const manual = { ...makeEntry('manual', '<manual/>'), disable_model_invocation: true };
		const normal = makeEntry('normal', '<normal/>');
		const snapshot = buildSkillRunSnapshot('/cwd', makeCatalog([manual, normal], '<inst/>'));

		expect(snapshot.total).toBe(1);
		expect(snapshot.entries.map((e) => e.name)).toEqual(['normal']);
		expect(snapshot.envelope).toContain('<normal/>');
		expect(snapshot.envelope).not.toContain('<manual/>');
		expect(snapshot.envelope).toContain('total="1"');
		expect(snapshot.envelope).toContain('included="1"');
		// the raw catalog is retained for the UI listing and the /skills picker
		expect(snapshot.catalog.skills).toHaveLength(2);
	});

	it('keeps entries whose flag is absent or false', () => {
		const normal = makeEntry('normal', '<normal/>');
		const legacy = { ...makeEntry('legacy', '<legacy/>'), disable_model_invocation: false };
		const snapshot = buildSkillRunSnapshot(undefined, makeCatalog([normal, legacy], '<inst/>'));

		expect(snapshot.total).toBe(2);
		expect(snapshot.entries.map((e) => e.name)).toEqual(['normal', 'legacy']);
		expect(snapshot.envelope).toContain('<normal/>');
		expect(snapshot.envelope).toContain('<legacy/>');
	});

	it('produces an empty model view when every entry is manual-only', () => {
		const manual = { ...makeEntry('manual', '<manual/>'), disable_model_invocation: true };
		const snapshot = buildSkillRunSnapshot('/cwd', makeCatalog([manual], '<inst/>'));

		expect(snapshot.total).toBe(0);
		expect(snapshot.entries).toEqual([]);
		expect(snapshot.envelope).toBe('<skills_catalog total="0" included="0"><inst/></skills_catalog>');
	});

	it('excludes locally disabled opaque IDs from entries, totals, the envelope, and budgeting', () => {
		const first = makeEntry('first', '<first/>');
		const second = makeEntry('second', '<second/>');
		const snapshot = buildSkillRunSnapshot(
			'/cwd',
			makeCatalog([first, second], '<inst/>'),
			new Set(['opaque-second'])
		);

		expect(snapshot.total).toBe(1);
		expect(snapshot.entries.map((e) => e.name)).toEqual(['first']);
		expect(snapshot.envelope).toContain('<first/>');
		expect(snapshot.envelope).toContain('total="1"');
		expect(snapshot.envelope).toContain('included="1"');
		expect(snapshot.envelope).not.toContain('<second/>');
		// The raw browsing catalog is untouched: disabled cards stay browsable.
		expect(snapshot.catalog.skills).toHaveLength(2);
	});

	it('combines locally disabled IDs with author-declared manual-only exclusion once', () => {
		const manual = { ...makeEntry('manual', '<manual/>'), disable_model_invocation: true };
		const disabled = makeEntry('disabled', '<disabled/>');
		const normal = makeEntry('normal', '<normal/>');
		const snapshot = buildSkillRunSnapshot(
			'/cwd',
			makeCatalog([manual, disabled, normal], '<inst/>'),
			new Set(['opaque-disabled'])
		);

		expect(snapshot.total).toBe(1);
		expect(snapshot.entries.map((e) => e.name)).toEqual(['normal']);
		expect(snapshot.envelope).toContain('<normal/>');
		expect(snapshot.envelope).not.toContain('<manual/>');
		expect(snapshot.envelope).not.toContain('<disabled/>');
	});

	it('leaves existing callers unaffected when no disabled set is passed', () => {
		// buildSkillRunSnapshot remains source-compatible: existing callers
		// pass no disabled set and see only manual-only exclusion.
		const manual = { ...makeEntry('manual', '<manual/>'), disable_model_invocation: true };
		const normal = makeEntry('normal', '<normal/>');
		const snapshot = buildSkillRunSnapshot('/cwd', makeCatalog([manual, normal], '<inst/>'));

		expect(snapshot.total).toBe(1);
		expect(snapshot.entries.map((e) => e.name)).toEqual(['normal']);
		expect(snapshot.catalog.skills).toHaveLength(2);
	});

	it('decides disabled exclusion by opaque ID, never by matching name', () => {
		// Two distinct opaque IDs share the same display name; disabling one
		// must never hide the other.
		const a = { ...makeEntry('duplicate', '<a/>'), id: 'opaque-a' };
		const b = { ...makeEntry('duplicate', '<b/>'), id: 'opaque-b' };
		const snapshot = buildSkillRunSnapshot(
			'/cwd',
			makeCatalog([a, b], '<inst/>'),
			new Set(['opaque-a'])
		);

		expect(snapshot.entries.map((e) => e.id)).toEqual(['opaque-b']);
	});
});

describe('SkillsPackingService.pack', () => {
	afterEach(() => {
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
	});

	const instructionXml = '<inst/>';
	const entryXmls = ['<e0/>', '<e1/>', '<e2/>'];

	function snapshot(): SkillRunSnapshot {
		return buildSkillRunSnapshot(
			'/w',
			makeCatalog(
				entryXmls.map((xml, i) => makeEntry(`s${i}`, xml)),
				instructionXml
			)
		);
	}

	function expectedEnvelope(included: number): string {
		const total = entryXmls.length;

		return `<skills_catalog total="${total}" included="${included}">${instructionXml}${entryXmls
			.slice(0, included)
			.join('')}</skills_catalog>`;
	}

	it('injects no envelope for a literal zero budget', async () => {
		const packed = await SkillsPackingService.pack(snapshot(), { budget: 0, mode: 'estimated' });

		expect(packed.envelope).toBe('');
		expect(packed.included).toBe(0);
		expect(packed.total).toBe(3);
	});

	it('injects no envelope for an empty catalog', async () => {
		const empty = buildSkillRunSnapshot(undefined, makeCatalog([], '<inst/>'));
		const packed = await SkillsPackingService.pack(empty, { budget: 10_000, mode: 'estimated' });

		expect(packed.envelope).toBe('');
		expect(packed.included).toBe(0);
		expect(packed.total).toBe(0);
	});

	it('returns the complete envelope when the budget fits everything', async () => {
		const snap = snapshot();
		const packed = await SkillsPackingService.pack(snap, { budget: 10_000, mode: 'estimated' });

		expect(packed.total).toBe(3);
		expect(packed.included).toBe(3);
		expect(packed.envelope).toBe(snap.envelope);
		expect(packed.estimated).toBe(true);
	});

	it('truncates entry fragments at the budget boundary, always keeping the instruction fragment', async () => {
		const budget = estimateSkillTokens(expectedEnvelope(2));
		const packed = await SkillsPackingService.pack(snapshot(), { budget, mode: 'estimated' });

		expect(packed.total).toBe(3);
		expect(packed.included).toBe(2);
		expect(packed.included).toBeLessThan(packed.total);
		expect(packed.envelope).toBe(expectedEnvelope(2));
		expect(packed.envelope).toContain(instructionXml);
		expect(packed.envelope).toContain('<e0/>');
		expect(packed.envelope).toContain('<e1/>');
		expect(packed.envelope).not.toContain('<e2/>');
	});

	it('keeps only the instruction fragment when no entry fits', async () => {
		const budget = estimateSkillTokens(expectedEnvelope(0));
		const packed = await SkillsPackingService.pack(snapshot(), { budget, mode: 'estimated' });

		expect(packed.included).toBe(0);
		expect(packed.envelope).toBe(expectedEnvelope(0));
		expect(packed.envelope).toContain(instructionXml);
	});

	it('never issues a tokenizer request in estimated mode', async () => {
		const fetchMock = vi.fn();

		vi.stubGlobal('fetch', fetchMock);

		const packed = await SkillsPackingService.pack(snapshot(), {
			budget: 10_000,
			mode: 'estimated'
		});

		expect(packed.estimated).toBe(true);
		expect(fetchMock).not.toHaveBeenCalled();
	});

	it('is deterministic: identical inputs produce identical packed envelopes', async () => {
		vi.stubGlobal('fetch', vi.fn());

		const snap = snapshot();
		const budget = estimateSkillTokens(expectedEnvelope(1));
		const first = await SkillsPackingService.pack(snap, { budget, mode: 'estimated' });
		const second = await SkillsPackingService.pack(snap, { budget, mode: 'estimated' });

		expect(second.envelope).toBe(first.envelope);
		expect(second.included).toBe(first.included);
		expect(second.estimated).toBe(first.estimated);
	});

	it('direct mode measures with the selected-model tokenizer request and flags', async () => {
		const tokenizer = charCountingTokenizer();

		vi.stubGlobal('fetch', tokenizer);

		const snap = snapshot();
		const packed = await SkillsPackingService.pack(snap, {
			budget: snap.envelope.length,
			mode: 'direct',
			model: 'selected-model'
		});

		expect(packed.estimated).toBe(false);
		expect(packed.included).toBe(packed.total);
		expect(packed.envelope).toBe(snap.envelope);
		expect(tokenizer).toHaveBeenCalled();

		const [url, init] = tokenizer.mock.calls[0];
		const body = JSON.parse(init.body as string) as Record<string, unknown>;

		expect(String(url)).toContain('/tokenize');
		expect(body).toMatchObject({
			add_special: false,
			content: snap.envelope,
			model: 'selected-model',
			parse_special: true
		});
	});

	it('direct mode packs to the tokenizer-measured boundary', async () => {
		vi.stubGlobal('fetch', charCountingTokenizer());

		const budget = expectedEnvelope(2).length;
		const packed = await SkillsPackingService.pack(snapshot(), {
			budget,
			mode: 'direct',
			model: 'selected-model'
		});

		expect(packed.estimated).toBe(false);
		expect(packed.included).toBe(2);
		expect(packed.envelope).toBe(expectedEnvelope(2));
	});

	it('falls back to a labeled estimate when the tokenizer request fails, without retrying', async () => {
		const fetchMock = vi.fn().mockRejectedValue(new Error('tokenizer unavailable'));

		vi.stubGlobal('fetch', fetchMock);

		const packed = await SkillsPackingService.pack(snapshot(), {
			budget: 10_000,
			mode: 'direct',
			model: 'selected-model'
		});

		expect(packed.estimated).toBe(true);
		expect(packed.envelope).toBe(snapshot().envelope);
		expect(packed.included).toBe(3);
		expect(fetchMock).toHaveBeenCalledTimes(1);
	});

	it('never uses a model when direct mode lacks a selected model', async () => {
		const fetchMock = vi.fn();

		vi.stubGlobal('fetch', fetchMock);

		const packed = await SkillsPackingService.pack(snapshot(), { budget: 10_000, mode: 'direct' });

		expect(packed.estimated).toBe(true);
		expect(fetchMock).not.toHaveBeenCalled();
	});

	it('retains the exact full-envelope token count for a complete direct pack', async () => {
		vi.stubGlobal('fetch', charCountingTokenizer());

		const snap = snapshot();
		const packed = await SkillsPackingService.pack(snap, {
			budget: 10_000,
			mode: 'direct',
			model: 'selected-model'
		});

		expect(packed.fullTokens).toBe(snap.envelope.length);
		expect(packed.included).toBe(snap.total);
		expect(packed.estimated).toBe(false);
	});

	it('retains the deterministic full-envelope estimate for a complete estimated pack', async () => {
		const snap = snapshot();
		const packed = await SkillsPackingService.pack(snap, { budget: 10_000, mode: 'estimated' });

		expect(packed.fullTokens).toBe(estimateSkillTokens(snap.envelope));
		expect(packed.included).toBe(snap.total);
		expect(packed.estimated).toBe(true);
	});

	it('returns null fullTokens for a zero budget without any tokenizer request', async () => {
		const fetchMock = vi.fn();

		vi.stubGlobal('fetch', fetchMock);

		const packed = await SkillsPackingService.pack(snapshot(), {
			budget: 0,
			mode: 'direct',
			model: 'selected-model'
		});

		expect(packed.fullTokens).toBeNull();
		expect(packed.included).toBe(0);
		expect(fetchMock).not.toHaveBeenCalled();
	});

	it('returns null fullTokens for an empty catalog without any tokenizer request', async () => {
		const fetchMock = vi.fn();

		vi.stubGlobal('fetch', fetchMock);

		const empty = buildSkillRunSnapshot(undefined, makeCatalog([], '<inst/>'));
		const packed = await SkillsPackingService.pack(empty, {
			budget: 10_000,
			mode: 'direct',
			model: 'selected-model'
		});

		expect(packed.fullTokens).toBeNull();
		expect(packed.included).toBe(0);
		expect(fetchMock).not.toHaveBeenCalled();
	});
});

describe('resolveSkillPackOptions', () => {
	it('resolves direct mode for a non-empty effective model in model mode', () => {
		expect(resolveSkillPackOptions('model-a', false, () => false)).toEqual({
			mode: 'direct',
			model: 'model-a'
		});
	});

	it('resolves estimated mode without an effective model', () => {
		expect(resolveSkillPackOptions('', false, () => false)).toEqual({ mode: 'estimated' });
	});

	it('resolves estimated mode in router mode when the model is not loaded', () => {
		expect(resolveSkillPackOptions('model-a', true, () => false)).toEqual({ mode: 'estimated' });
	});

	it('resolves direct mode in router mode only for a loaded model', () => {
		expect(resolveSkillPackOptions('model-a', true, (model) => model === 'model-a')).toEqual({
			mode: 'direct',
			model: 'model-a'
		});
	});
});

// Guards the persisted maxSkillBudget contract, including zero and sanitization.
describe('maxSkillBudget registry', () => {
	it('defaults to 2000 in the persisted config defaults', () => {
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.MAX_SKILL_BUDGET]).toBe(2000);
	});

	it('is registered as a non-negative integer input control in the agentic section', () => {
		const section = SETTINGS_CHAT_SECTIONS.find((s) => s.slug === SETTINGS_SECTION_SLUGS.AGENTIC);

		expect(section).toBeDefined();

		const field = section?.fields?.find((f) => f.key === SETTINGS_KEYS.MAX_SKILL_BUDGET);

		expect(field).toMatchObject({
			isPositiveInteger: true,
			min: 0,
			type: SettingsFieldType.INPUT
		});
	});

	it('participates in the save-time numeric clamp list', () => {
		expect(POSITIVE_INTEGER_FIELDS).toContain(SETTINGS_KEYS.MAX_SKILL_BUDGET);
	});
});

describe('normalizeSkillBudget', () => {
	it('keeps a valid non-negative integer untouched', () => {
		expect(normalizeSkillBudget(2000)).toBe(2000);
		expect(normalizeSkillBudget(0)).toBe(0);
	});

	it('clamps negative values to zero', () => {
		expect(normalizeSkillBudget(-5)).toBe(0);
		expect(normalizeSkillBudget(-0.1)).toBe(0);
	});

	it('rounds fractional values to integers', () => {
		expect(normalizeSkillBudget(3.7)).toBe(4);
		expect(normalizeSkillBudget(2500.2)).toBe(2500);
	});

	it('falls back to the default for non-numeric or non-finite values', () => {
		expect(normalizeSkillBudget('2500')).toBe(2000);
		expect(normalizeSkillBudget(undefined)).toBe(2000);
		expect(normalizeSkillBudget(null)).toBe(2000);
		expect(normalizeSkillBudget(Number.NaN)).toBe(2000);
		expect(normalizeSkillBudget(Number.POSITIVE_INFINITY)).toBe(2000);
	});
});