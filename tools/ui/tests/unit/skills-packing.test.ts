// Guards budget packing, snapshot immutability, envelope serialization, and budget settings.

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
import type { SkillRunSnapshot } from '$lib/types';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { jsonResponse, makeCatalog, makeEntry } from '../fixtures/skills';

const INSTRUCTION_XML = '<inst/>';
const ENTRY_XMLS = ['<e0/>', '<e1/>', '<e2/>'];

function snapshot(): SkillRunSnapshot {
	return buildSkillRunSnapshot(
		'/w',
		makeCatalog(
			ENTRY_XMLS.map((xml, i) => makeEntry(`s${i}`, { catalog_xml: xml })),
			INSTRUCTION_XML
		)
	);
}

function expectedEnvelope(included: number): string {
	return `<skills_catalog total="${ENTRY_XMLS.length}" included="${included}">${INSTRUCTION_XML}${ENTRY_XMLS.slice(0, included).join('')}</skills_catalog>`;
}

/** Deterministic tokenizer double: one token per character. */
function charCountingTokenizer() {
	return vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
		const body = JSON.parse(init?.body as string) as { content: string };

		return jsonResponse({ tokens: Array.from({ length: body.content.length }, (_, i) => i) });
	});
}

describe('estimateSkillTokens', () => {
	it('computes the ceiling of UTF-8 bytes / 4, counting bytes not characters', () => {
		expect(estimateSkillTokens('')).toBe(0);
		expect(estimateSkillTokens('x'.repeat(10))).toBe(3);
		expect(estimateSkillTokens('x'.repeat(13))).toBe(4);
		// U+20AC is 3 UTF-8 bytes: 12 bytes -> 3 tokens.
		expect(estimateSkillTokens('\u20ac'.repeat(4))).toBe(3);
	});
});

describe('envelope serialization and run snapshots', () => {
	it('serializes the complete envelope deterministically, preserving server XML verbatim', () => {
		const raw = '<skill a="1">&amp;<b>&lt;raw&gt; &quot;text&quot;</b></skill>';
		const catalog = makeCatalog([makeEntry('escaped', { catalog_xml: raw })], '<inst/>');

		expect(serializeSkillCatalogEnvelope(catalog)).toBe(
			'<skills_catalog total="1" included="1"><inst/><skill a="1">&amp;<b>&lt;raw&gt; &quot;text&quot;</b></skill></skills_catalog>'
		);
	});

	it('copies entries immutably and freezes them so store mutations cannot reach the snapshot', () => {
		const manual = { ...makeEntry('manual', { catalog_xml: '<manual/>' }), disable_model_invocation: true };
		const normal = makeEntry('normal', { catalog_xml: '<normal/>' });
		const catalog = makeCatalog([manual, normal], '<inst/>');
		const snapshot = buildSkillRunSnapshot('/cwd', catalog, new Set(['opaque-normal']));

		catalog.skills[0].catalog_xml = 'MUTATED';
		catalog.catalog_instruction_xml = 'MUTATED';
		catalog.skills.push(makeEntry('late'));

		// Manual-only and locally disabled entries drop out of the model view
		// by opaque ID, never by name; the raw browsing catalog is retained.
		expect(snapshot.total).toBe(0);
		expect(snapshot.entries).toEqual([]);
		expect(snapshot.envelope).toBe('<skills_catalog total="0" included="0"><inst/></skills_catalog>');
		expect(snapshot.catalog.skills).toHaveLength(3);
		expect(Object.isFrozen(snapshot.entries)).toBe(true);
		expect(Object.isFrozen(snapshot.entries[0])).toBe(true);
	});

	it('excludes locally disabled entries by opaque ID, never by matching name', () => {
		const a = { ...makeEntry('duplicate', { catalog_xml: '<a/>' }), id: 'opaque-a' };
		const b = { ...makeEntry('duplicate', { catalog_xml: '<b/>' }), id: 'opaque-b' };
		const snapshot = buildSkillRunSnapshot('/cwd', makeCatalog([a, b], '<inst/>'), new Set(['opaque-a']));

		expect(snapshot.entries.map((e) => e.id)).toEqual(['opaque-b']);
		expect(snapshot.envelope).toContain('<b/>');
		expect(snapshot.envelope).not.toContain('<a/>');
	});
});

describe('SkillsPackingService.pack', () => {
	afterEach(() => {
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
	});

	it('injects no envelope and no token count for a zero budget or an empty catalog', async () => {
		const fetchMock = vi.fn();

		vi.stubGlobal('fetch', fetchMock);

		const zero = await SkillsPackingService.pack(snapshot(), {
			budget: 0,
			mode: 'direct',
			model: 'selected-model'
		});
		const empty = await SkillsPackingService.pack(
			buildSkillRunSnapshot(undefined, makeCatalog([], '<inst/>')),
			{ budget: 10_000, mode: 'direct', model: 'selected-model' }
		);

		for (const packed of [zero, empty]) {
			expect(packed.envelope).toBe('');
			expect(packed.included).toBe(0);
			expect(packed.fullTokens).toBeNull();
		}

		expect(zero.total).toBe(3);
		expect(empty.total).toBe(0);
		expect(fetchMock).not.toHaveBeenCalled();
	});

	it('returns the complete envelope and the exact full token count when the budget fits', async () => {
		vi.stubGlobal('fetch', charCountingTokenizer());

		const snap = snapshot();
		const packed = await SkillsPackingService.pack(snap, {
			budget: 10_000,
			mode: 'direct',
			model: 'selected-model'
		});

		expect(packed).toMatchObject({
			envelope: snap.envelope,
			estimated: false,
			fullTokens: snap.envelope.length,
			included: 3,
			total: 3
		});

		const [url, init] = vi.mocked(fetch).mock.calls[0];
		const body = JSON.parse(init.body as string) as Record<string, unknown>;

		expect(String(url)).toContain('/tokenize');
		expect(body).toMatchObject({
			add_special: false,
			content: snap.envelope,
			model: 'selected-model',
			parse_special: true
		});
	});

	it('truncates at the budget boundary in both modes, always keeping the instruction fragment', async () => {
		const estimated = await SkillsPackingService.pack(snapshot(), {
			budget: estimateSkillTokens(expectedEnvelope(2)),
			mode: 'estimated'
		});

		vi.stubGlobal('fetch', charCountingTokenizer());

		const direct = await SkillsPackingService.pack(snapshot(), {
			budget: expectedEnvelope(2).length,
			mode: 'direct',
			model: 'selected-model'
		});
		const instructionOnly = await SkillsPackingService.pack(snapshot(), {
			budget: estimateSkillTokens(expectedEnvelope(0)),
			mode: 'estimated'
		});

		for (const packed of [estimated, direct]) {
			expect(packed.included).toBe(2);
			expect(packed.envelope).toBe(expectedEnvelope(2));
		}

		expect(estimated.estimated).toBe(true);
		expect(direct.estimated).toBe(false);
		expect(instructionOnly.included).toBe(0);
		expect(instructionOnly.envelope).toBe(expectedEnvelope(0));
	});

	it('never issues a tokenizer request in estimated mode or direct mode without a model', async () => {
		const fetchMock = vi.fn();

		vi.stubGlobal('fetch', fetchMock);

		await SkillsPackingService.pack(snapshot(), { budget: 10_000, mode: 'estimated' });
		await SkillsPackingService.pack(snapshot(), { budget: 10_000, mode: 'direct' });

		expect(fetchMock).not.toHaveBeenCalled();
	});

	it('falls back to a labeled estimate without retrying when the tokenizer request fails', async () => {
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

	it('retains the deterministic full-envelope estimate for a complete estimated pack', async () => {
		const snap = snapshot();
		const packed = await SkillsPackingService.pack(snap, { budget: 10_000, mode: 'estimated' });

		expect(packed.fullTokens).toBe(estimateSkillTokens(snap.envelope));
		expect(packed.estimated).toBe(true);
	});
});

describe('resolveSkillPackOptions', () => {
	it.each([
		['model mode with a model', 'model-a', false, () => false, { mode: 'direct', model: 'model-a' }],
		['no effective model', '', false, () => false, { mode: 'estimated' }],
		['router mode with an unloaded model', 'model-a', true, () => false, { mode: 'estimated' }],
		[
			'router mode with a loaded model',
			'model-a',
			true,
			(model: string) => model === 'model-a',
			{ mode: 'direct', model: 'model-a' }
		]
	] as const)('resolves %s', (_label, model, routerMode, isModelLoaded, expected) => {
		expect(resolveSkillPackOptions(model, routerMode, isModelLoaded)).toEqual(expected);
	});
});

describe('maxSkillBudget settings contract', () => {
	it('defaults to 2000, is a clamped non-negative integer field, and participates in save-time clamping', () => {
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.MAX_SKILL_BUDGET]).toBe(2000);

		const section = SETTINGS_CHAT_SECTIONS.find((s) => s.slug === SETTINGS_SECTION_SLUGS.AGENTIC);
		const field = section?.fields?.find((f) => f.key === SETTINGS_KEYS.MAX_SKILL_BUDGET);

		expect(field).toMatchObject({
			isPositiveInteger: true,
			min: 0,
			type: SettingsFieldType.INPUT
		});
		expect(POSITIVE_INTEGER_FIELDS).toContain(SETTINGS_KEYS.MAX_SKILL_BUDGET);
	});

	it('normalizes persisted values: valid kept, negatives clamped, fractions rounded, junk defaulted', () => {
		expect(normalizeSkillBudget(2000)).toBe(2000);
		expect(normalizeSkillBudget(0)).toBe(0);
		expect(normalizeSkillBudget(-5)).toBe(0);
		expect(normalizeSkillBudget(3.7)).toBe(4);
		expect(normalizeSkillBudget('2500')).toBe(2000);
		expect(normalizeSkillBudget(Number.NaN)).toBe(2000);
	});
});
