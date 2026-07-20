// Tier 1 perf harness for the agentic thread.
//
// Drives ChatMessageAgenticContent the way the real streaming pipeline does:
// chat.svelte.ts -> conversations.svelte.ts:184 replaces the message object per
// SSE chunk (`{ ...old, ...updates }`), which changes prop identity and cascades
// through deriveAgenticSections into every tool-call block in the message.
//
// The fixture is parameterized so the *scaling curve* identifies the culprit -
// a single number would not. See tests/client/README-perf.md.
//
// Run: npx vitest --project=client --run tests/client/agentic-stream.perf.svelte.test.ts

import { describe, it } from 'vitest';
import { render } from 'vitest-browser-svelte';
import { tick } from 'svelte';
import AgenticPerfWrapper from './components/AgenticPerfWrapper.svelte';
import ChatMessagesPerfWrapper from './components/ChatMessagesPerfWrapper.svelte';
import { perfState } from './components/agentic-perf-state.svelte';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import type { DatabaseMessage } from '$lib/types';
import { MessageRole } from '$lib/enums';

// --- fixture construction -------------------------------------------------

interface FixtureOpts {
	/** Completed tool-call sections preceding the streaming text. */
	priorToolCalls: number;
	/** Size of each tool result blob. */
	toolResultBytes: number;
	/** Number of edit_file calls (each a 400x400-line diff). */
	editFileEdits: number;
	/** Leave an unclosed ``` fence at the end of the streamed content. */
	openCodeFence: boolean;
	/**
	 * Emit a blank line every N chunks so the content forms real markdown
	 * blocks. 0 = one unbroken paragraph, which defeats MarkdownContent's
	 * stable-block cache entirely (worst case). Typical prose has breaks.
	 */
	paragraphEvery: number;
}

const DEFAULTS: FixtureOpts = {
	priorToolCalls: 0,
	toolResultBytes: 1024,
	editFileEdits: 0,
	openCodeFence: false,
	paragraphEvery: 0
};

function blob(bytes: number, seed: string): string {
	const line = `${seed} output line with some representative width to it`;
	const n = Math.max(1, Math.ceil(bytes / (line.length + 1)));
	const out: string[] = [];
	for (let i = 0; i < n; i++) out.push(`${line} ${i}`);
	return out.join('\n');
}

function diffLines(n: number, seed: string): string {
	const out: string[] = [];
	for (let i = 0; i < n; i++) out.push(`${seed} line ${i} const value_${i} = compute(${i});`);
	return out.join('\n');
}

let msgSeq = 0;

function baseMessage(overrides: Partial<DatabaseMessage>): DatabaseMessage {
	return {
		id: `m${msgSeq++}`,
		convId: 'perf-conv',
		type: 'text',
		timestamp: 0,
		role: MessageRole.ASSISTANT,
		content: '',
		parent: null,
		children: [],
		...overrides
	} as DatabaseMessage;
}

function buildFixture(opts: FixtureOpts): {
	message: DatabaseMessage;
	toolMessages: DatabaseMessage[];
} {
	const toolCalls: unknown[] = [];
	const toolMessages: DatabaseMessage[] = [];

	for (let i = 0; i < opts.priorToolCalls; i++) {
		const id = `call_${i}`;
		toolCalls.push({
			id,
			type: 'function',
			function: {
				name: 'exec_shell_command',
				arguments: JSON.stringify({ command: `grep -rn "thing_${i}" src/` })
			}
		});
		toolMessages.push(
			baseMessage({
				role: MessageRole.TOOL,
				toolCallId: id,
				content: `${blob(opts.toolResultBytes, `t${i}`)}\n[exit code: 0]`
			})
		);
	}

	for (let i = 0; i < opts.editFileEdits; i++) {
		const id = `edit_${i}`;
		toolCalls.push({
			id,
			type: 'function',
			function: {
				name: 'edit_file',
				arguments: JSON.stringify({
					path: `/src/file_${i}.ts`,
					edits: [{ old_text: diffLines(400, 'old'), new_text: diffLines(400, 'new') }]
				})
			}
		});
		toolMessages.push(
			baseMessage({
				role: MessageRole.TOOL,
				toolCallId: id,
				content: JSON.stringify({ result: 'ok', edits_applied: 1 })
			})
		);
	}

	const message = baseMessage({
		toolCalls: toolCalls.length > 0 ? JSON.stringify(toolCalls) : undefined,
		content: ''
	});

	return { message, toolMessages };
}

// --- the driver -----------------------------------------------------------

interface Sample {
	label: string;
	tokens: number;
	mean: number;
	p95: number;
	max: number;
	/** Sum of the synchronous per-token windows. */
	total: number;
	/** Wall-clock for the whole run incl. deferred rAF work, then settle. */
	wall: number;
}

function nextFrame(): Promise<void> {
	return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

const results: Sample[] = [];

/**
 * Replays `tokens` streamed chunks, replacing the message object each time
 * exactly as conversations.svelte.ts:184 does, flushing Svelte and forcing
 * layout so the measurement includes render + style/layout, not just script.
 */
async function measure(label: string, partial: Partial<FixtureOpts>, tokens = 60) {
	const opts = { ...DEFAULTS, ...partial };
	const { message, toolMessages } = buildFixture(opts);

	perfState.message = message;
	perfState.toolMessages = toolMessages;
	perfState.isStreaming = true;

	// Every wrapper reads the same module state, so a leftover mount from a
	// previous fixture would re-render on each mutation and fold its cost into
	// this measurement. Tear down explicitly between fixtures.
	const { unmount } = render(AgenticPerfWrapper);

	await tick();

	const CHUNK = 'The quick brown fox jumps over the lazy dog. ';
	let accumulated = opts.openCodeFence ? '```notalanguage\n' : '';
	const durations: number[] = [];

	const wallStart = performance.now();

	for (let i = 0; i < tokens; i++) {
		accumulated += CHUNK;
		if (opts.paragraphEvery > 0 && (i + 1) % opts.paragraphEvery === 0) {
			accumulated += '\n\n';
		}

		const t0 = performance.now();
		// Mirrors updateMessageAtIndex: a brand-new object identity per chunk.
		perfState.message = { ...perfState.message!, content: accumulated };
		await tick();
		void document.body.offsetHeight; // force style + layout
		durations.push(performance.now() - t0);

		// MarkdownContent coalesces its parse into a rAF, so that work lands
		// outside the window above. Yield a frame each iteration so it is
		// captured in `wall` - the gap between `wall` and `total` is the
		// deferred cost.
		await nextFrame();
	}

	// Let any trailing coalesced work drain before stopping the clock.
	await nextFrame();
	await nextFrame();
	const wall = performance.now() - wallStart;

	await unmount();

	durations.sort((a, b) => a - b);
	const total = durations.reduce((a, b) => a + b, 0);

	results.push({
		label,
		tokens,
		mean: total / durations.length,
		p95: durations[Math.floor(durations.length * 0.95)],
		max: durations[durations.length - 1],
		total,
		wall
	});
}

function report() {
	const pad = (s: string, n: number) => s.padEnd(n);
	const num = (n: number) => n.toFixed(2).padStart(8);

	const header = `${pad('fixture', 40)}${pad('tok', 5)}${'mean'.padStart(8)}${'p95'.padStart(8)}${'max'.padStart(8)}${'sync'.padStart(9)}${'wall'.padStart(9)}`;
	const lines = [
		'',
		'=== Tier 1: ms per streamed token (agentic content subtree) ===',
		'mean/p95/max = synchronous window per token (script + style + layout).',
		'sync  = sum of those windows.',
		'wall  = whole run, incl. work MarkdownContent defers into its own rAF.',
		'        NOTE: wall carries a ~16.7ms/token idle floor from the harness',
		'        yielding a frame each iteration (60 tokens => ~1000ms floor).',
		'        Compare wall ACROSS fixtures / against the baseline row,',
		'        never against sync.',
		'',
		header,
		'-'.repeat(header.length)
	];

	for (const r of results) {
		lines.push(
			`${pad(r.label, 40)}${pad(String(r.tokens), 5)}${num(r.mean)}${num(r.p95)}${num(r.max)}${num(r.total)}${num(r.wall)}`
		);
	}

	lines.push('');
	console.log(lines.join('\n'));
}

// --- conversation-level driver --------------------------------------------
// The per-message driver above cannot see the fan-out in ChatMessages: a single
// token mutation invalidates `displayMessages`, which rebuilds a fresh
// toolMessages array for EVERY message in the conversation. Drive the real
// store through the real list component to measure that.

async function measureConversation(
	label: string,
	priorMessages: number,
	tokens = 60,
	/**
	 * Give each prior assistant turn a resolved tool call, so the fixture pays
	 * `hasAgenticContent`'s JSON.parse and the tool-message grouping walk that a
	 * real agent thread would - plain prose messages skip both.
	 */
	agentic = false
) {
	const history: DatabaseMessage[] = [];
	for (let i = 0; i < priorMessages; i++) {
		const isAssistant = i % 2 !== 0;

		if (isAssistant && agentic) {
			const id = `prior_call_${i}`;
			history.push(
				baseMessage({
					role: MessageRole.ASSISTANT,
					content: `Message ${i}`,
					toolCalls: JSON.stringify([
						{
							id,
							type: 'function',
							function: {
								name: 'exec_shell_command',
								arguments: JSON.stringify({ command: `grep -rn "thing_${i}" src/` })
							}
						}
					])
				})
			);
			history.push(
				baseMessage({
					role: MessageRole.TOOL,
					toolCallId: id,
					content: `${blob(1024, `r${i}`)}\n[exit code: 0]`
				})
			);
			continue;
		}

		history.push(
			baseMessage({
				role: isAssistant ? MessageRole.ASSISTANT : MessageRole.USER,
				content: `Message ${i}: ${blob(512, `m${i}`)}`
			})
		);
	}

	const streaming = baseMessage({ role: MessageRole.ASSISTANT, content: '' });
	history.push(streaming);

	conversationsStore.activeMessages = history;

	const { unmount } = render(ChatMessagesPerfWrapper);
	await tick();

	const idx = conversationsStore.findMessageIndex(streaming.id);
	const CHUNK = 'The quick brown fox jumps over the lazy dog. ';
	let accumulated = '';
	const durations: number[] = [];
	const wallStart = performance.now();

	for (let i = 0; i < tokens; i++) {
		accumulated += CHUNK;

		const t0 = performance.now();
		// The real path: chat.svelte.ts -> conversations.svelte.ts.
		conversationsStore.updateMessageAtIndex(idx, { content: accumulated });
		await tick();
		void document.body.offsetHeight;
		durations.push(performance.now() - t0);

		await nextFrame();
	}

	await nextFrame();
	await nextFrame();
	const wall = performance.now() - wallStart;

	await unmount();
	conversationsStore.activeMessages = [];

	durations.sort((a, b) => a - b);
	const total = durations.reduce((a, b) => a + b, 0);

	results.push({
		label,
		tokens,
		mean: total / durations.length,
		p95: durations[Math.floor(durations.length * 0.95)],
		max: durations[durations.length - 1],
		total,
		wall
	});
}

// --- the matrix -----------------------------------------------------------
// Sequential, in one test, so the table prints together and the samples do not
// interleave with other suites competing for the main thread.

describe('agentic streaming perf', () => {
	it('scales', { timeout: 600_000 }, async () => {
		// Baseline: plain text streaming, nothing agentic.
		await measure('baseline: no tool calls', {});

		// Knob: priorToolCalls. Flat => no fan-out. Linear => fan-out confirmed.
		await measure('priorToolCalls=1  (1KB results)', { priorToolCalls: 1 });
		await measure('priorToolCalls=5  (1KB results)', { priorToolCalls: 5 });
		await measure('priorToolCalls=20 (1KB results)', { priorToolCalls: 20 });

		// Knob: toolResultBytes, at fixed section count.
		await measure('5 calls x 200KB results', {
			priorToolCalls: 5,
			toolResultBytes: 200 * 1024
		});

		// Knob: editFileEdits (computeLineDiff, 400x400 LCS).
		await measure('3 edit_file calls (400x400 diff)', { editFileEdits: 3 });

		// Knob: openCodeFence (hljs highlightAuto on partial code).
		await measure('open code fence, unknown language', { openCodeFence: true });

		// Conversation level: does streaming one message cost more as the
		// conversation grows? Flat => no fan-out. Linear => confirmed.
		await measureConversation('convo: 1 prior message', 1);
		await measureConversation('convo: 10 prior messages', 10);
		await measureConversation('convo: 40 prior messages', 40);

		// Same, but each prior assistant turn carries a resolved tool call.
		await measureConversation('convo: 10 prior, agentic', 10, 60, true);
		await measureConversation('convo: 40 prior, agentic', 40, 60, true);

		// Message length: MarkdownContent re-parses the whole accumulated string
		// each frame, so per-token cost should climb as the response grows.
		// A rising mean across these three rows means O(n^2) over the stream.
		// One unbroken paragraph: worst case, stable-block cache never applies.
		await measure('len 60tok  1-para (worst case)', {}, 60);
		await measure('len 250tok 1-para (worst case)', {}, 250);
		await measure('len 600tok 1-para (worst case)', {}, 600);

		// Same lengths, broken into paragraphs every 8 chunks: the typical shape,
		// where only the trailing paragraph should be unstable.
		await measure('len 60tok  paras (typical)', { paragraphEvery: 8 }, 60);
		await measure('len 250tok paras (typical)', { paragraphEvery: 8 }, 250);
		await measure('len 600tok paras (typical)', { paragraphEvery: 8 }, 600);

		report();
	});
});
