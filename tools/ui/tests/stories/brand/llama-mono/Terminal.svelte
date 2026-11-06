<!--
	Llama Mono in the terminal: llama-server starting on a dark window, and a llama-cli chat on a light one, with a
	shell prompt that carries the Llama icon at U+E000, one cell wide like every other character. Colour is kept for
	what it says (the prompt, a warning, a success), and cinder only marks the cursor.
-->
<script lang="ts">
	const ICON = '';

	type Line = [string, string][]; // a line's segments: [class, text]

	const prompt = (dir: string, branch: string, command: string): Line => [
		['ps1', `${ICON} ${dir} `],
		['branch', branch ? `${branch} ` : ''],
		['ps1', '$ '],
		['cmd', command]
	];

	// llama-server starting
	const server: Line[] = [
		prompt(
			'~/models',
			'main',
			'llama-server -hf ggml-org/gemma-3-12b-it-GGUF --port 9931 -c 32768'
		),
		[['dim', 'build: 6123 (a1b2c3d4) with Apple clang 17.0.0 for arm64-apple-darwin']],
		[['dim', 'system info: n_threads = 12 / 16 | Metal : EMBED_LIBRARY = 1']],
		[['', 'load_tensors: offloaded 49/49 layers to GPU']],
		[['', 'load_tensors:   Metal_Mapped model buffer size =  7685.49 MiB']],
		[['', 'llama_context: n_ctx = 32768, n_batch = 2048, flash_attn = auto']],
		[['warn', 'warn: the chat template has no system role, merging it into the first turn']],
		[['', 'srv  load_model: initializing slots, n_slots = 4']],
		[['ok', 'main: server is listening on http://127.0.0.1:9931']],
		[['ok', 'main: starting the main loop...']]
	];

	// a llama-cli chat, ending on the cursor
	const cli: Line[] = [
		prompt('~/models', '', 'llama-cli -hf ggml-org/SmolLM3-3B-GGUF'),
		[['dim', '== Running in interactive mode. ==']],
		[['dim', ' - Press Ctrl+C to interject at any time.']],
		[['dim', ' - Press Return to return control to the AI.']],
		[],
		[['you', "> What's a GGUF file, in one sentence?"]],
		[['', "A GGUF file holds a model's weights and everything needed to run it"]],
		[['', '(architecture, tokenizer, chat template) in one memory-mappable file.']],
		[],
		[['dim', '[ Prompt: 312.4 t/s | Generation: 96.1 t/s ]']],
		[],
		[['you', '> ']]
	];
</script>

<div class="terminals">
	<figure class="window dark">
		<figcaption>
			<span class="dots"><i></i><i></i><i></i></span>llama-server · zsh · 100×24
		</figcaption>

		<!-- prettier-ignore -->
		<pre class="screen">{#each server as line, i (i)}{#each line as [c, t], j (j)}<span class={c}>{t}</span>{/each}
{/each}</pre>
	</figure>

	<figure class="window light">
		<figcaption><span class="dots"><i></i><i></i><i></i></span>llama-cli · zsh · 100×24</figcaption>

		<!-- prettier-ignore -->
		<pre class="screen">{#each cli as line, i (i)}{#each line as [c, t], j (j)}<span class={c}>{t}</span>{/each}{#if i === cli.length - 1}<i class="cursor"></i>{/if}
{/each}</pre>
	</figure>
</div>

<div class="notes">
	<p>
		<b>One cell each.</b> The icon in the prompt, the box-drawing in a TUI and every digit take the same
		600 units, so a log column, a progress bar or a table of numbers stays in line however it is filled.
	</p>

	<p>
		<b>Weight for structure.</b> The command is SemiBold, its output Regular, and what's only context
		(the build, the system info) steps back in grey, so the eye finds the command first.
	</p>

	<p>
		<b>Colour for meaning.</b> Yellow is a warning and green is good news; cinder is the one accent, and
		here it's only the cursor.
	</p>
</div>

<style>
	/* one above the other: a server's lines run long */
	.terminals {
		display: grid;
		gap: 16px;
	}
	.window {
		margin: 0;
		overflow: hidden;
		border-radius: 10px;
	}
	.window.dark {
		--bg: #111111;
		--fg: #fafafa;
		--dim: #8a8a8a;
		--bar: #1c1c1c;
		--ok: #4ade80;
		--warn: #facc15;
		--you: #93c5fd;
	}
	.window.light {
		--bg: #fafafa;
		--fg: #111111;
		--dim: #737373;
		--bar: #ececec;
		--ok: #15803d;
		--warn: #a16207;
		--you: #1d4ed8;
		box-shadow: inset 0 0 0 1px rgb(0 0 0 / 0.1);
	}
	figcaption {
		display: flex;
		align-items: center;
		gap: 12px;
		padding: 10px 14px;
		background: var(--bar);
		color: var(--dim);
		font: 500 12px var(--code);
	}
	.dots {
		display: flex;
		gap: 6px;
	}
	.dots i {
		width: 10px;
		height: 10px;
		border-radius: 50%;
		background: var(--dim);
		opacity: 0.5;
	}
	.screen {
		margin: 0;
		padding: 16px 18px 20px;
		overflow-x: auto;
		background: var(--bg);
		color: var(--fg);
		font: 400 13px / 1.6 var(--code);
	}
	.ps1 {
		font-weight: 600;
		color: var(--dim);
	}
	.cmd {
		font-weight: 600;
	}
	.branch {
		color: var(--you);
	}
	.dim {
		color: var(--dim);
	}
	.ok {
		color: var(--ok);
		font-weight: 500;
	}
	.warn {
		color: var(--warn);
	}
	.you {
		color: var(--you);
		font-weight: 600;
	}
	.cursor {
		display: inline-block;
		width: 1ch;
		height: 1.15em;
		vertical-align: -0.2em;
		background: var(--cinder);
	}

	.notes {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr));
		gap: 16px 40px;
	}
	.notes p {
		margin: 0;
		font: 400 15px / 1.5 var(--ui);
		color: var(--muted);
	}
	.notes b {
		font-weight: 600;
		color: var(--ink);
	}
</style>
