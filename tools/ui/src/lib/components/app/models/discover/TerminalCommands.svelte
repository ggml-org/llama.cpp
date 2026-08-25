<script lang="ts">
	import { Check, Copy, Server, SquareTerminal } from '@lucide/svelte';
	import { type DraftVariant } from '$lib/constants';
	import { copyToClipboard } from '$lib/utils';

	interface Props {
		/** Full HuggingFace model id, The draft sidecar sits in the same repo. */
		modelId: string;
		/** Draft sidecar variants present in the repo (mtp, dflash, dspark, eagle3). */
		draftVariants?: DraftVariant[];
	}

	let { draftVariants = [], modelId }: Props = $props();

	// llama.cpp --spec-type value for each draft variant.
	const SPEC_TYPE: Record<DraftVariant, string> = {
		dflash: 'draft-dflash',
		dspark: 'draft-dspark',
		eagle3: 'eagle3',
		mmproj: '',
		mtp: 'draft-mtp'
	};

	let copiedIndex = $state<number | null>(null);

	async function handleCopy(index: number, text: string) {
		await copyToClipboard(text);
		copiedIndex = index;
		setTimeout(() => (copiedIndex = null), 1500);
	}

	// One box per binary (serve / cli). Each box lists one command per available
	// draft sidecar, or just the base command when none is present.
	let boxes = $derived.by(() => {
		const variants = draftVariants.filter((v) => v !== 'mmproj');
		const build = (bin: string) => {
			const base = `llama ${bin} -hf ${modelId}`;

			if (variants.length === 0) return [base];

			return variants.map((v) => `${base} -hfd ${modelId} --spec-type ${SPEC_TYPE[v]}`);
		};

		return [
			{ commands: build('serve'), icon: Server, title: 'Serve' },
			{ commands: build('cli'), icon: SquareTerminal, title: 'CLI' }
		];
	});
</script>

<div class="space-y-3">
	{#each boxes as box (box.title)}
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

			<div class="space-y-1 p-2">
				{#each box.commands as cmd, i (cmd)}
					<div
						class="group flex items-center justify-between gap-2 rounded px-2 py-1 font-mono text-xs"
					>
						<span class="truncate text-foreground/90">{cmd}</span>

						<button
							aria-label="Copy command"
							class="shrink-0 text-muted-foreground/60 transition-colors hover:text-foreground"
							onclick={() => handleCopy(i, cmd)}
							type="button"
						>
							{#if copiedIndex === i}
								<Check class="h-3.5 w-3.5 text-green-500" />
							{:else}
								<Copy class="h-3.5 w-3.5" />
							{/if}
						</button>
					</div>
				{/each}
			</div>
		</div>
	{/each}
</div>
