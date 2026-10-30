<script lang="ts">
	import { Check, Copy } from '@lucide/svelte';
	import * as Select from '$lib/components/ui/select';
	import { COPY_FEEDBACK_RESET_MS, MODEL_ID, SERVE_COMMAND, SPEC_TYPE } from '$lib/constants';
	import type { ModelDraftSidecar } from '$lib/enums';
	import { copyToClipboard } from '$lib/utils';

	interface Props {
		/** Full HuggingFace repo id, e.g. `ggml-org/gemma-3-4b-it-GGUF`. */
		modelId: string;
		/** Quant the command pins on `-hf`; null when the row's file carries none. */
		quant?: string | null;
		/**
		 * Draft kind the panel downloads with the model, null when the sidecars are
		 * off. The server resolves the draft file itself, so the command only names
		 * its type.
		 */
		draftKind?: ModelDraftSidecar | null;
		/** Draft kinds the repo ships; the command offers them in its own pick. */
		draftKinds?: ModelDraftSidecar[];
		/** Fires when the command's pick changes, so the panel's own pick follows. */
		onDraftKindChange?: (kind: ModelDraftSidecar) => void;
	}

	let {
		draftKind = null,
		draftKinds = [],
		modelId,
		onDraftKindChange,
		quant = null
	}: Props = $props();

	/** The llama serve command: the row's quant and, when it rides along, the draft. */
	let command = $derived.by(() => {
		const parts = [
			SERVE_COMMAND.BIN,
			SERVE_COMMAND.SUBCOMMAND,
			SERVE_COMMAND.MODEL_FLAG,
			quant ? `${modelId}${MODEL_ID.QUANTIZATION_SEPARATOR}${quant}` : modelId
		];

		if (draftKind) {
			parts.push(SERVE_COMMAND.SPEC_TYPE_FLAG, SPEC_TYPE[draftKind]);
		}

		return parts.join(' ');
	});

	let copied = $state(false);

	async function copy() {
		await copyToClipboard(command);
		copied = true;
		setTimeout(() => (copied = false), COPY_FEEDBACK_RESET_MS);
	}
</script>

<div
	class="relative flex items-center gap-2 overflow-hidden rounded-lg border border-border/40 bg-background py-2.5 pl-4 pr-10 shadow-xs dark:border-border/35 dark:bg-background/50"
>
	<!-- Single line: long commands scroll horizontally instead of wrapping. -->
	<div
		class="flex min-w-0 flex-1 items-center gap-x-2 overflow-x-auto py-0.5 font-mono text-xs whitespace-nowrap text-foreground/90"
	>
		<span class="shrink-0">{SERVE_COMMAND.BIN}</span>

		<span class="shrink-0">{SERVE_COMMAND.SUBCOMMAND}</span>

		<span class="shrink-0">{SERVE_COMMAND.MODEL_FLAG}</span>

		<span class="shrink-0">
			{modelId}{quant ? MODEL_ID.QUANTIZATION_SEPARATOR : ''}
		</span>

		<!-- the row's own quant, so no pick is needed here -->
		{#if quant}
			<span class="shrink-0 rounded-md border border-primary/15 bg-primary/[0.07] px-1.5">
				{quant}
			</span>
		{/if}

		<!-- The draft rides with the model, so the flag only names its type; the
		     checkbox below turns it on, the server resolves the file. The pick is
		     the panel's own, so changing it here changes it there. -->
		{#if draftKind}
			<span class="shrink-0">{SERVE_COMMAND.SPEC_TYPE_FLAG}</span>

			<!-- the select only earns its chrome when there is a real choice to make -->
			{#if draftKinds.length > 1}
				<Select.Root
					onValueChange={(value) => value && onDraftKindChange?.(value as ModelDraftSidecar)}
					type="single"
					value={draftKind}
				>
					<Select.Trigger
						aria-label="Draft type"
						class="border-primary/15 bg-primary/[0.07] font-mono text-foreground hover:bg-primary/15 focus-visible:border-primary/40 focus-visible:ring-0"
						size="xs"
					>
						{SPEC_TYPE[draftKind]}
					</Select.Trigger>

					<Select.Content class="font-mono text-xs">
						{#each draftKinds as kind (kind)}
							<Select.Item class="text-xs" label={SPEC_TYPE[kind]} value={kind}>
								{SPEC_TYPE[kind]}
							</Select.Item>
						{/each}
					</Select.Content>
				</Select.Root>
			{:else}
				<span class="shrink-0 rounded-md border border-primary/15 bg-primary/[0.07] px-1.5">
					{SPEC_TYPE[draftKind]}
				</span>
			{/if}
		{/if}
	</div>

	<button
		aria-label="Copy command"
		class="absolute top-1/2 right-2 -translate-y-1/2 cursor-pointer rounded-md p-1.5 text-muted-foreground/70 transition-colors hover:bg-primary/10 hover:text-foreground"
		onclick={copy}
		type="button"
	>
		{#if copied}
			<Check class="size-icon-sm text-fg-success" />
		{:else}
			<Copy class="size-icon-sm" />
		{/if}
	</button>
</div>
