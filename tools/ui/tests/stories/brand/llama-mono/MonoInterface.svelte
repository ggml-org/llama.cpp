<!--
	Llama Mono in the user interface, beside Llama Sans: the design system's own components with llama.cpp's content.
	Mono takes what a person types, copies or compares (code, commands, model files and quants, endpoints, live
	numbers, shortcuts); Sans keeps the prose and the labels.
-->
<script lang="ts">
	import { Check, Copy, Gauge } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';

	let endpoint = $state('http://127.0.0.1:9931/v1');
	let seed = $state('42');
	let temperature = $state('0.8');
	let copied = $state(false);

	const command = `llama-server -hf ggml-org/gemma-3-12b-it-GGUF \\
    --n-gpu-layers 99 --ctx-size 32768 --port 9931`;

	const response = `{
  "model": "gemma-3-12b-it-Q4_K_M.gguf",
  "usage": { "prompt_tokens": 52, "completion_tokens": 512 },
  "timings": { "predicted_per_second": 38.41 }
}`;

	const shortcuts = [
		{ keys: ['⌘', 'K'], label: 'Search conversations' },
		{ keys: ['⇧', '⌘', 'O'], label: 'New chat' },
		{ keys: ['⌘', '.'], label: 'Stop generating' }
	];

	function copy() {
		copied = true;
		setTimeout(() => (copied = false), 1200);
	}
</script>

<div class="bg-background px-4 py-10 text-foreground sm:px-8">
	<div class="mx-auto grid max-w-5xl gap-8">
		<header class="grid gap-2">
			<p class="font-mono text-caption tracking-widest text-muted-foreground uppercase">
				Llama Mono · beside Llama Sans
			</p>

			<h1 class="text-4xl font-bold tracking-tight">User Interface</h1>

			<p class="max-w-[62ch] text-body text-muted-foreground">
				The design system's own components, at their real sizes. Llama Mono takes what a person
				types, copies or compares: code and commands, model files and quants, endpoints, live
				numbers and shortcuts. Llama Sans keeps the prose and the labels, so the two read as one
				family.
			</p>
		</header>

		<div class="grid gap-4 md:grid-cols-2">
			<!-- a reply in the chat: Sans prose, inline code and a code block in Mono -->
			<Card.Root class="md:col-span-2">
				<Card.Content class="grid gap-3 text-body">
					<p>
						Run it with every layer on the GPU: <code class="inline">--n-gpu-layers 99</code>
						offloads all that fits, and <code class="inline">--ctx-size</code> sets how many tokens it
						keeps in view.
					</p>

					<div class="overflow-hidden rounded-lg border bg-muted/40">
						<div
							class="flex items-center justify-between border-b px-3 py-1.5 text-caption text-muted-foreground"
						>
							<span class="font-mono">bash</span>

							<Button onclick={copy} size="sm" variant="ghost">
								{#if copied}<Check /> Copied{:else}<Copy /> Copy{/if}
							</Button>
						</div>

						<pre class="block">{command}</pre>
					</div>

					<p class="font-mono text-caption text-muted-foreground tabular-nums">
						512 tokens · 13.3 s · 38.4 t/s
					</p>
				</Card.Content>
			</Card.Root>

			<!-- a model: its name in Sans, what identifies it in Mono -->
			<Card.Root>
				<Card.Header>
					<Card.Title>Gemma 3 12B Instruct</Card.Title>

					<Card.Description class="font-mono">ggml-org/gemma-3-12b-it-GGUF</Card.Description>
				</Card.Header>

				<Card.Content class="grid gap-4">
					<div class="flex flex-wrap gap-2">
						<Badge class="font-mono" variant="secondary">Q4_K_M</Badge>

						<Badge class="font-mono" variant="secondary">7.3 GB</Badge>

						<Badge class="font-mono" variant="secondary">ctx 131072</Badge>

						<Badge variant="success">Loaded</Badge>
					</div>

					<dl class="grid grid-cols-3 gap-3">
						{#each [['Prompt', '312.4', 't/s'], ['Generation', '38.4', 't/s'], ['Memory', '7.9', 'GB']] as [label, value, unit] (label)}
							<div class="grid gap-0.5">
								<dt class="text-caption text-muted-foreground">{label}</dt>

								<dd class="font-mono text-xl font-semibold tabular-nums">
									{value}<span class="ml-1 text-caption font-normal text-muted-foreground"
										>{unit}</span
									>
								</dd>
							</div>
						{/each}
					</dl>
				</Card.Content>

				<Card.Footer class="gap-2">
					<Gauge class="size-4 text-muted-foreground" />

					<span class="truncate font-mono text-caption text-muted-foreground"
						>gemma-3-12b-it-Q4_K_M.gguf</span
					>
				</Card.Footer>
			</Card.Root>

			<!-- settings: the labels in Sans, the values a person types in Mono -->
			<Card.Root>
				<Card.Header>
					<Card.Title>Connection</Card.Title>

					<Card.Description>Any OpenAI-compatible endpoint</Card.Description>
				</Card.Header>

				<Card.Content class="grid gap-4">
					<div class="grid gap-2">
						<Label for="mono-endpoint">Base URL</Label>

						<Input bind:value={endpoint} class="font-mono" id="mono-endpoint" />
					</div>

					<div class="grid grid-cols-2 gap-3">
						<div class="grid gap-2">
							<Label for="mono-temp">Temperature</Label>

							<Input bind:value={temperature} class="font-mono tabular-nums" id="mono-temp" />
						</div>

						<div class="grid gap-2">
							<Label for="mono-seed">Seed</Label>

							<Input bind:value={seed} class="font-mono tabular-nums" id="mono-seed" />
						</div>
					</div>

					<ul class="grid gap-2">
						{#each shortcuts as s (s.label)}
							<li class="flex items-center justify-between gap-4 text-label-sm">
								<span>{s.label}</span>

								<span class="flex gap-1">
									{#each s.keys as k (k)}<kbd>{k}</kbd>{/each}
								</span>
							</li>
						{/each}
					</ul>
				</Card.Content>
			</Card.Root>

			<!-- an API response, as the web UI's developer view shows it -->
			<Card.Root class="md:col-span-2">
				<Card.Header>
					<Card.Title>Response</Card.Title>

					<Card.Description class="font-mono"
						>POST /v1/chat/completions · 200 · 13.4 s</Card.Description
					>
				</Card.Header>

				<Card.Content>
					<pre class="block rounded-lg bg-muted/40">{response}</pre>
				</Card.Content>
			</Card.Root>
		</div>
	</div>
</div>

<style>
	pre.block {
		margin: 0;
		padding: 12px 14px;
		overflow-x: auto;
		font: 400 13px / 1.6 var(--font-mono);
	}
	code.inline {
		padding: 0.1em 0.35em;
		border-radius: 4px;
		background: var(--muted);
		font: 500 0.9em var(--font-mono);
	}
	kbd {
		display: inline-grid;
		place-items: center;
		min-width: 1.6rem;
		height: 1.5rem;
		padding: 0 0.35rem;
		border: 1px solid var(--border);
		border-radius: 5px;
		background: var(--background);
		font: 500 12px var(--font-mono);
		color: var(--muted-foreground);
	}
</style>
