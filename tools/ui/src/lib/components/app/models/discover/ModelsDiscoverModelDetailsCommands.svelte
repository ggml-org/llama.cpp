<script lang="ts">
	import { Check, Copy, Server, SquareTerminal } from '@lucide/svelte';
	import { isAuxSidecar, type ModelSidecar } from '$lib/constants';
	import { ModelAuxSidecar, ModelDraftSidecar } from '$lib/enums';
	import { copyToClipboard } from '$lib/utils';

	interface Props {
		/** Full HuggingFace repo id, e.g. `ggml-org/gemma-3-4b-it-GGUF`. */
		modelId: string;
		/** Available quantization tags, e.g. `Q4_K_M`, `Q8_0`. */
		quants: string[];
		/** Draft sidecars present in the repo (mtp, dflash, dspark, eagle3). */
		sidecars?: ModelSidecar[];
	}

	let { modelId, quants, sidecars = [] }: Props = $props();

	let pickedQuant = $state<string | null>(null);
	let selectedQuant = $derived(pickedQuant ?? quants[0] ?? null);
	let selectedSidecar = $state<ModelSidecar | null>(null);

	// llama.cpp --spec-type value for each draft sidecar.
	const SPEC_TYPE: Record<ModelSidecar, string> = {
		[ModelAuxSidecar.MMPROJ]: '',
		[ModelDraftSidecar.DFLASH]: 'draft-dflash',
		[ModelDraftSidecar.DSPARK]: 'draft-dspark',
		[ModelDraftSidecar.EAGLE3]: 'eagle3',
		[ModelDraftSidecar.MTP]: 'draft-mtp'
	};

	let copiedIndex = $state<number | null>(null);

	async function handleCopy(index: number, text: string) {
		await copyToClipboard(text);
		copiedIndex = index;
		setTimeout(() => (copiedIndex = null), 1500);
	}

	// One box per binary (serve / cli). The command embeds inline selectors for
	// the quant and the draft sidecar type.
	let boxes = $derived.by(() => {
		const quant = selectedQuant ?? quants[0];

		if (!quant) return [];

		const draft = selectedSidecar && !isAuxSidecar(selectedSidecar) ? selectedSidecar : null;
		const specType = draft ? `--spec-type ${SPEC_TYPE[draft]}` : '';
		const build = (bin: string) => {
			const parts = ['llama', bin, '-hfd', modelId];

			if (specType) parts.push(specType);

			return parts.join(' ');
		};

		return [
			{ command: build('serve'), icon: Server, title: 'Serve' },
			{ command: build('cli'), icon: SquareTerminal, title: 'CLI' }
		];
	});
</script>

<div class="space-y-3">
	<div class="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
		<span>Quant</span>

		<select
			bind:value={selectedQuant}
			class="rounded border bg-background px-1.5 py-0.5 font-mono text-xs"
		>
			{#each quants as quant (quant)}
				<option value={quant}>{quant}</option>
			{/each}
		</select>

		{#if sidecars.length}
			<span>Draft</span>

			<select
				bind:value={selectedSidecar}
				class="rounded border bg-background px-1.5 py-0.5 font-mono text-xs"
			>
				<option value={null}>none</option>

				{#each sidecars as sidecar (sidecar)}
					<option value={sidecar}>{sidecar}</option>
				{/each}
			</select>
		{/if}
	</div>

	{#each boxes as box, i (box.title)}
		<div
			class="overflow-hidden rounded-md"
			style="background: var(--code-background); border: 1px solid color-mix(in oklch, var(--border) 30%, transparent);"
		>
			<div
				class="flex items-center gap-2 px-3 py-2"
				style="border-bottom: 1px solid color-mix(in oklch, var(--border) 30%, transparent);"
			>
				<box.icon class="h-3.5 w-3.5 text-muted-foreground/60" />

				<span class="text-xs font-medium text-foreground/80">{box.title}</span>
			</div>

			<div class="flex items-center justify-between gap-2 p-2">
				<span class="truncate font-mono text-xs text-foreground/90">{box.command}</span>

				<button
					aria-label="Copy command"
					class="shrink-0 text-muted-foreground/60 transition-colors hover:text-foreground"
					onclick={() => handleCopy(i, box.command)}
					type="button"
				>
					{#if copiedIndex === i}
						<Check class="h-3.5 w-3.5 text-green-500" />
					{:else}
						<Copy class="h-3.5 w-3.5" />
					{/if}
				</button>
			</div>
		</div>
	{/each}
</div>
