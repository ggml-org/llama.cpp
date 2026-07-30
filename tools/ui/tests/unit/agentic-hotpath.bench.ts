// Microbenchmarks for the functions on the agentic-thread streaming hot path.
//
// Every function here is invoked from a `$derived` that is invalidated on each
// streamed token, for every tool-call section in the message. These numbers give
// the per-call cost; multiply by (tokens x sections) for the real damage.
//
// Run: npx vitest bench --project=unit tests/unit/agentic-hotpath.bench.ts

import { bench, describe } from 'vitest';

import { remark } from 'remark';
import remarkBreaks from 'remark-breaks';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import rehypeStringify from 'rehype-stringify';
import { all as lowlightAll } from 'lowlight';

import { computeLineDiff } from '$lib/utils/compute-line-diff';
import { extractSearchResults, extractSearchQuery } from '$lib/utils/search-results';
import { parsePartialJsonArgs } from '$lib/utils/parse-partial-json-args';
import { highlightCode, detectIncompleteCodeBlock } from '$lib/utils/code';
import { classifyToolResult, parseToolResultWithImages } from '$lib/utils/agentic';
import { preprocessLaTeX } from '$lib/utils/latex-protection';

// --- fixtures -------------------------------------------------------------

function lines(n: number, seed: string): string {
	const out: string[] = [];
	for (let i = 0; i < n; i++) out.push(`${seed} line ${i} const value_${i} = compute(${i});`);
	return out.join('\n');
}

const SHELL_OUTPUT_1KB = lines(12, 'out');
const SHELL_OUTPUT_200KB = lines(2600, 'out');
const SHELL_OUTPUT_2MB = lines(26000, 'out');

// Realistic exec_shell_command result: the exit-code marker is the final line,
// which is what the un-anchored EXIT_CODE regex has to scan the whole blob for.
const SHELL_2MB_WITH_EXIT = `${SHELL_OUTPUT_2MB}\n[exit code: 0]`;

const EDIT_OLD_400 = lines(400, 'old');
const EDIT_NEW_400 = lines(400, 'new');
const EDIT_OLD_50 = lines(50, 'old');
const EDIT_NEW_50 = lines(50, 'new');

const WRITE_FILE_ARGS = JSON.stringify({
	path: '/src/lib/thing.ts',
	content: lines(1500, 'src')
});

const MARKDOWN_50KB = Array.from(
	{ length: 400 },
	(_, i) =>
		`## Section ${i}\n\nSome **bold** prose with a [link](https://example.com) and \`inline\` code.\n\n- bullet one\n- bullet two\n\n\`\`\`ts\nconst x${i} = ${i};\n\`\`\`\n`
).join('\n');

const CODE_BLOCK_5KB = lines(60, 'code');

// --- A: computeLineDiff (O(m*n) LCS, allocates a full matrix) --------------

describe('computeLineDiff', () => {
	bench('50 x 50 lines', () => {
		computeLineDiff(EDIT_OLD_50, EDIT_NEW_50);
	});

	bench('400 x 400 lines', () => {
		computeLineDiff(EDIT_OLD_400, EDIT_NEW_400);
	});
});

// --- B: the unified processor, rebuilt per call ---------------------------
// Mirrors MarkdownContent.svelte:144-176. The local rehype/remark plugins are
// omitted (they pull in browser-only modules); rehypeHighlight + lowlightAll is
// the dominant term, so this is a lower bound on the real cost.

function buildProcessor() {
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let proc: any = remark().use(remarkGfm);
	proc = proc.use(remarkMath).use(remarkBreaks).use(remarkRehype).use(rehypeKatex);
	return proc.use(rehypeHighlight, { languages: lowlightAll }).use(rehypeStringify, {
		allowDangerousHtml: true
	});
}

describe('markdown processor', () => {
	bench('build processor (per processMarkdown call, x2)', () => {
		buildProcessor();
	});

	const prebuilt = buildProcessor();

	bench('parse 50KB markdown with a prebuilt processor', () => {
		prebuilt.parse(MARKDOWN_50KB);
	});
});

// Isolates the O(n^2) term measured in Tier 1: processMarkdown re-parses the
// WHOLE accumulated string every frame, while only the last block can have
// changed. If parse() dominates at these sizes, incremental parsing (re-parsing
// only the tail after the last stable block) is the fix; if not, the cost is
// downstream in transform/stringify/DOM.
describe('markdown parse scaling (whole-string reparse per frame)', () => {
	const proc = buildProcessor();
	const prose = (kb: number) =>
		Array.from(
			{ length: Math.ceil((kb * 1024) / 64) },
			(_, i) => `The quick brown fox jumps over the lazy dog. Sentence ${i}.`
		).join(' ');

	const MD_3KB = prose(3);
	const MD_11KB = prose(11);
	const MD_26KB = prose(26);

	bench('parse 3KB', () => {
		proc.parse(MD_3KB);
	});

	bench('parse 11KB', () => {
		proc.parse(MD_11KB);
	});

	bench('parse 26KB', () => {
		proc.parse(MD_26KB);
	});

	bench('parse only a 200-char tail (proposed incremental)', () => {
		proc.parse(MD_26KB.slice(-200));
	});
});

// processMarkdown runs these over the WHOLE accumulated string every frame too,
// before parse() even starts. Measure them before assuming parse is the target.
describe('other whole-string passes per frame', () => {
	const prose = (kb: number) =>
		Array.from(
			{ length: Math.ceil((kb * 1024) / 64) },
			(_, i) => `The quick brown fox jumps over the lazy dog. Sentence ${i}.`
		).join(' ');

	const MD_3KB = prose(3);
	const MD_11KB = prose(11);
	const MD_26KB = prose(26);
	const MD_26KB_LATEX = `${MD_26KB} and some math $x^2 + y^2 = z^2$ inline.`;

	bench('preprocessLaTeX 3KB (no latex present)', () => {
		preprocessLaTeX(MD_3KB);
	});

	bench('preprocessLaTeX 11KB (no latex present)', () => {
		preprocessLaTeX(MD_11KB);
	});

	bench('preprocessLaTeX 26KB (no latex present)', () => {
		preprocessLaTeX(MD_26KB);
	});

	bench('preprocessLaTeX 26KB (latex present)', () => {
		preprocessLaTeX(MD_26KB_LATEX);
	});

	bench('detectIncompleteCodeBlock 26KB', () => {
		detectIncompleteCodeBlock(MD_26KB);
	});
});

// --- C: extractSearchResults, run for EVERY tool of every type ------------

describe('extractSearchResults (only .length > 0 is consumed)', () => {
	bench('1KB non-search tool result', () => {
		extractSearchResults(SHELL_OUTPUT_1KB);
	});

	bench('200KB non-search tool result', () => {
		extractSearchResults(SHELL_OUTPUT_200KB);
	});

	bench('2MB non-search tool result', () => {
		extractSearchResults(SHELL_OUTPUT_2MB);
	});
});

describe('extractSearchQuery', () => {
	bench('write_file args (throws on the JSON.parse path)', () => {
		extractSearchQuery(WRITE_FILE_ARGS);
	});
});

// --- E: the un-anchored exit-code regex -----------------------------------
// Reproduced inline so the bench is independent of the current call site.

const EXIT_CODE_TAIL = /\[exit code: (-?\d+)\]\s*$/;

describe('exit-code regex', () => {
	bench('whole 2MB blob (current behaviour)', () => {
		SHELL_2MB_WITH_EXIT.match(EXIT_CODE_TAIL);
	});

	bench('last 64 chars only (proposed)', () => {
		SHELL_2MB_WITH_EXIT.slice(-64).match(EXIT_CODE_TAIL);
	});
});

// --- per-line result parsers ----------------------------------------------

describe('parseToolResultWithImages', () => {
	bench('1KB', () => {
		parseToolResultWithImages(SHELL_OUTPUT_1KB, []);
	});

	bench('200KB', () => {
		parseToolResultWithImages(SHELL_OUTPUT_200KB, []);
	});

	bench('2MB', () => {
		parseToolResultWithImages(SHELL_OUTPUT_2MB, []);
	});
});

describe('classifyToolResult', () => {
	bench('200KB plain text (falls through to looksLikeMarkdown)', () => {
		classifyToolResult(SHELL_OUTPUT_200KB);
	});
});

describe('parsePartialJsonArgs', () => {
	bench('write_file args, ~60KB (char-by-char scan)', () => {
		parsePartialJsonArgs(WRITE_FILE_ARGS);
	});
});

// --- F / I: highlight.js ---------------------------------------------------

describe('highlightCode', () => {
	bench('short bash command (title row, per token)', () => {
		highlightCode('grep -rn "foo" src/ | head -50', 'bash');
	});

	bench('5KB known language', () => {
		highlightCode(CODE_BLOCK_5KB, 'typescript');
	});

	bench('5KB unknown language -> highlightAuto', () => {
		highlightCode(CODE_BLOCK_5KB, 'not-a-language');
	});
});
