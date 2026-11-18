/**
 * Synthetic conversations for demos, in the UI's export format (llama.app JSONL
 * traces): a `session` record followed by one `message` record per message.
 * Deterministic, so a trace and the demos recorded from it don't drift. Messages
 * carry no model, so the UI doesn't flag one the server doesn't have.
 */

import { createHash } from 'node:crypto';

const HARNESS = 'llama.app';
const START = Date.UTC(2026, 9, 1, 9, 0, 0);

interface ToolCall {
	function: { arguments: string; name: string };
	id: string;
	type: 'function';
}

interface TraceMessage {
	children: string[];
	content: string;
	convId: string;
	id: string;
	parent: null | string;
	reasoningContent?: string;
	role: 'assistant' | 'system' | 'tool' | 'user';
	timestamp: number;
	toolCallId?: string;
	toolCalls?: ToolCall[];
	type: 'root' | 'text';
}

/** Builds one linear conversation, message by message. */
class Trace {
	private calls = 0;
	private readonly id: string;
	private readonly messages: TraceMessage[] = [];
	private readonly name: string;

	assistant(content: string, reasoning?: string): this {
		return this.push({ content, reasoningContent: reasoning, role: 'assistant' });
	}

	constructor(name: string) {
		this.name = name;
		this.id = uuid(name, 'conv');
		this.push({ content: '', role: 'system', type: 'root' });
	}

	toJsonl(): string {
		const last = this.messages.at(-1)!;
		const session = {
			currNode: last.id,
			harness: HARNESS,
			id: this.id,
			lastModified: last.timestamp,
			name: this.name,
			type: 'session'
		};

		return [session, ...this.messages.map((message) => ({ message, type: 'message' }))]
			.map((record) => JSON.stringify(record))
			.join('\n');
	}

	/** An assistant turn calling one tool, followed by the tool's result. */
	tool(name: string, args: Record<string, unknown>, result: string, reasoning?: string): this {
		const id = uuid(this.name, `call-${this.calls++}`).replaceAll('-', '');

		this.push({
			content: '',
			reasoningContent: reasoning,
			role: 'assistant',
			toolCalls: [{ function: { arguments: JSON.stringify(args), name }, id, type: 'function' }]
		});

		return this.push({ content: result, role: 'tool', toolCallId: id });
	}

	user(content: string): this {
		return this.push({ content, role: 'user' });
	}

	private push(message: Partial<TraceMessage> & Pick<TraceMessage, 'content' | 'role'>): this {
		const parent = this.messages.at(-1);
		const index = this.messages.length;
		const entry: TraceMessage = {
			children: [],
			convId: this.id,
			id: uuid(this.name, `msg-${index}`),
			parent: parent?.id ?? null,
			timestamp: START + index * 20_000,
			type: 'text',
			...message
		};

		parent?.children.push(entry.id);
		this.messages.push(entry);

		return this;
	}
}

function uuid(...parts: string[]): string {
	const hex = createHash('sha1').update(parts.join(':')).digest('hex');

	return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-4${hex.slice(13, 16)}-a${hex.slice(17, 20)}-${hex.slice(20, 32)}`;
}

const STEPS = [
	['set up the project', 'cargo new forecast && cd forecast', 'Cargo.toml', 'toml'],
	['parse command line arguments', 'cargo add clap --features derive', 'src/cli.rs', 'rust'],
	['call the weather API', 'cargo add reqwest --features json,blocking', 'src/api.rs', 'rust'],
	['deserialize the response', 'cargo add serde --features derive', 'src/model.rs', 'rust'],
	['cache responses on disk', 'cargo add directories', 'src/cache.rs', 'rust'],
	['print a table', 'cargo add comfy-table', 'src/table.rs', 'rust'],
	['handle errors', 'cargo add anyhow', 'src/main.rs', 'rust'],
	['add colors', 'cargo add owo-colors', 'src/style.rs', 'rust'],
	['read a config file', 'cargo add toml', 'src/config.rs', 'rust'],
	['write tests', 'cargo add --dev assert_cmd predicates', 'tests/cli.rs', 'rust'],
	['set up CI', 'mkdir -p .github/workflows', '.github/workflows/ci.yml', 'yaml'],
	['publish a release', 'cargo install cargo-dist', 'dist-workspace.toml', 'toml']
] as const;

function codeFor(step: number, file: string, lang: string): string {
	const lines =
		lang === 'rust'
			? [
					`// ${file}`,
					'use anyhow::{Context, Result};',
					'',
					`pub fn step_${step}(input: &str) -> Result<Vec<String>> {`,
					'    let mut out = Vec::new();',
					'    for (i, line) in input.lines().enumerate() {',
					'        let trimmed = line.trim();',
					'        if trimmed.is_empty() {',
					'            continue;',
					'        }',
					'        out.push(format!("{i:>3}: {trimmed}"));',
					'    }',
					`    Ok(out).context("step ${step} failed")`,
					'}'
				]
			: lang === 'yaml'
				? [
						`# ${file}`,
						'on: [push, pull_request]',
						'jobs:',
						'  test:',
						'    runs-on: ubuntu-latest'
					]
				: [`# ${file}`, `name = "forecast"`, `version = "0.${step}.0"`, 'edition = "2024"'];

	return `\`\`\`${lang}\n${lines.join('\n')}\n\`\`\``;
}

/** A long chat with tall messages, for scrolling and landing demos. */
function longChat(): Trace {
	const trace = new Trace('Building a weather CLI in Rust');

	for (let i = 0; i < 36; i++) {
		const [goal, command, file, lang] = STEPS[i % STEPS.length];
		const round = Math.floor(i / STEPS.length);
		const ask =
			round === 0 ? `How do I ${goal}?` : `Can we revisit how we ${goal}? (pass ${round + 1})`;

		trace.user(ask);
		trace.assistant(
			[
				`To ${goal}, start from a clean working tree so the change stays easy to review.`,
				'',
				'```sh',
				command,
				'```',
				'',
				`Then add \`${file}\`:`,
				'',
				codeFor(i + 1, file, lang),
				'',
				'A few things to keep in mind:',
				'',
				`- Keep \`${file}\` focused on one job, so the next step only touches what it needs.`,
				'- Run `cargo clippy` after each step; it catches most mistakes early.',
				'- Commit once it builds, so you can always go back to a working version.',
				'',
				i % 3 === 0
					? `Next we can look at how to ${STEPS[(i + 1) % STEPS.length][0]}.`
					: 'Let me know if anything here does not build.'
			].join('\n'),
			`The user wants to ${goal}. Give the command, the file and a short checklist.`
		);
	}

	return trace;
}

function testLog(failing: boolean): string {
	const lines = ['running 48 tests'];

	for (let i = 1; i <= 48; i++) {
		const ok = !failing || (i !== 17 && i !== 31);

		lines.push(
			`test cache::tests::entry_${String(i).padStart(2, '0')} ... ${ok ? 'ok' : 'FAILED'}`
		);
	}

	if (failing) {
		lines.push(
			'',
			'failures:',
			'',
			'---- cache::tests::entry_17 stdout ----',
			"thread 'cache::tests::entry_17' panicked at src/cache.rs:88:9:",
			'assertion `left == right` failed: entry expired too early',
			'  left: Some(1759312800)',
			' right: Some(1759316400)',
			'',
			'---- cache::tests::entry_31 stdout ----',
			"thread 'cache::tests::entry_31' panicked at src/cache.rs:88:9:",
			'assertion `left == right` failed: entry expired too early',
			'',
			'test result: FAILED. 46 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out'
		);
	} else {
		lines.push('', 'test result: ok. 48 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out');
	}

	return `${lines.join('\n')}\n\n[exit code: ${failing ? 101 : 0}]`;
}

function reasoning(topic: string, paragraphs: number): string {
	const out: string[] = [];

	for (let i = 0; i < paragraphs; i++) {
		out.push(
			`${i === 0 ? `Looking at ${topic}.` : 'Another angle:'} The cache stores an expiry timestamp per entry, ` +
				'and the failing tests are the ones created right at an hour boundary. If the expiry is computed ' +
				'from local time while the check uses UTC, entries created near the boundary would expire an hour ' +
				`early. Step ${i + 1}: confirm by reading how \`expires_at\` is built, then compare with \`is_fresh\`.`
		);
	}

	return out.join('\n\n');
}

/** An agentic session with long reasoning, shell output and code, for nested scroller demos. */
function agenticDebugging(): Trace {
	const trace = new Trace('Fix the flaky cache tests');
	const fixed = Array.from({ length: 60 }, (_, i) =>
		i === 0
			? 'use chrono::{DateTime, Duration, Utc};'
			: i === 30
				? '        let expires_at = Utc::now() + self.ttl; // was Local::now()'
				: `        // cache.rs line ${i + 1}: entry handling stays the same`
	).join('\n');

	trace.user('The cache tests fail on CI every few runs. Can you find out why and fix it?');

	for (let round = 0; round < 3; round++) {
		trace.tool(
			'exec_shell_command',
			{ command: 'cargo test cache -- --test-threads=1' },
			testLog(true),
			reasoning('the failing cache tests', 4 + round)
		);
		trace.tool(
			'exec_shell_command',
			{ command: "grep -n 'expires_at\\|is_fresh' src/cache.rs" },
			Array.from(
				{ length: 30 },
				(_, i) => `${80 + i}:        let expires_at = Local::now() + self.ttl; // entry ${i}`
			).join('\n') + '\n\n[exit code: 0]',
			reasoning('where the expiry is computed', 3)
		);
		trace.assistant(
			[
				`Round ${round + 1}: the expiry uses local time while the freshness check uses UTC.`,
				'',
				'```rust',
				fixed,
				'```',
				'',
				'I switched the expiry to `Utc::now()` so both sides use the same clock.'
			].join('\n'),
			reasoning('the fix', 5)
		);
		trace.user(round < 2 ? 'Still flaky on CI, can you check again?' : 'Run the tests once more.');
	}

	trace.tool(
		'exec_shell_command',
		{ command: 'cargo test cache' },
		testLog(false),
		reasoning('the final test run', 2)
	);
	trace.assistant(
		'All 48 cache tests pass now, including the two that failed at the hour boundary.'
	);

	return trace;
}

/** Built-in traces, by name. */
export const DEMO_TRACES: Record<string, () => Trace> = {
	'agentic-debugging': agenticDebugging,
	'long-chat': longChat
};

export function generateTrace(name: string): null | string {
	return DEMO_TRACES[name]?.().toJsonl() ?? null;
}
