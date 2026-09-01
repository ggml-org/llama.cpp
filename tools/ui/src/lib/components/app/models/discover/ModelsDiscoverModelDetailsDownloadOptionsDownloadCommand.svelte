<script lang="ts">
	import { Check, Copy } from '@lucide/svelte';
	import { type ModelSidecar } from '$lib/constants';
	import { copyToClipboard } from '$lib/utils';
	import { SELECT_CLASS, type QuantOption } from './download-options.utils';

	interface Props {
		modelId: string;
		/** Full command text, copied to the clipboard as-is. */
		command: string;
		baseOptions: QuantOption[];
		draftOptions: (QuantOption & { badge: ModelSidecar | null; shared: boolean })[];
		/** Value of the base quant select, mirrored by the parent. */
		basePick: string;
		/** Value of the draft quant select; empty when no draft is picked. */
		draftPick: string;
		/** Quant after `-hf`; null when the base file carries no quant. */
		mainQuant: string | null;
		/** True when the base quant is a deliberate pick, not the default preview. */
		mainSelected: boolean;
		/** Quant after `-hfd`, shown only when a draft is picked. */
		draftQuant: string | null;
		/** `--spec-type` value; null hides the draft segment. */
		specType: string | null;
		onBasePick: (path: string) => void;
		onDraftPick: (path: string) => void;
	}

	let {
		baseOptions,
		basePick,
		command,
		draftOptions,
		draftPick,
		draftQuant,
		mainQuant,
		mainSelected,
		modelId,
		onBasePick,
		onDraftPick,
		specType
	}: Props = $props();

	let copied = $state(false);

	async function copy() {
		await copyToClipboard(command);
		copied = true;
		setTimeout(() => (copied = false), 1500);
	}
</script>

<div
	class="relative flex items-center gap-2 overflow-hidden rounded-lg border border-border/40 bg-background py-2.5 pl-4 pr-10 shadow-xs dark:border-border/35 dark:bg-background/50"
>
	<div
		aria-hidden="true"
		class="pointer-events-none absolute inset-y-0 left-0 w-px bg-primary/20"
	></div>

	<span aria-hidden="true" class="shrink-0 select-none font-mono text-xs text-muted-foreground/40"
		>$</span
	>

	<!-- Single line: long commands scroll horizontally instead of wrapping. -->
	<div
		class="flex min-w-0 flex-1 items-center gap-x-2 overflow-x-auto py-0.5 font-mono text-xs whitespace-nowrap text-foreground/90"
	>
		<span class="shrink-0">llama</span>

		<span class="shrink-0">serve</span>

		<span class="shrink-0">-hf</span>

		<span class="shrink-0">{modelId}{mainQuant ? ':' : ''}</span>

		<!-- Base quant: always part of the command, the 8-bit file by default. -->
		{#if baseOptions.length}
			<select
				aria-label="Base model quantization"
				class="{SELECT_CLASS} {mainSelected ? '' : 'border-dashed'} -ml-2"
				onchange={(e) => onBasePick(e.currentTarget.value)}
				title={mainSelected ? undefined : 'Default quant - pick a file above or another quant here'}
				value={basePick}
			>
				{#each baseOptions as option (option.path)}
					<option disabled={option.disabled} value={option.path}>
						{option.label}
					</option>
				{/each}
			</select>
		{/if}

		<!-- Draft segment: appears once a draft is picked, quant inline too. -->
		{#if specType !== null}
			<span>-hfd</span>

			<span class="shrink-0">{modelId}{draftQuant ? ':' : ''}</span>

			<select
				aria-label="Draft model quantization"
				class={SELECT_CLASS}
				onchange={(e) => onDraftPick(e.currentTarget.value)}
				value={draftPick}
			>
				{#each draftOptions as option (option.path)}
					<option disabled={option.disabled} value={option.path}>
						{#if option.shared}
							{option.badge?.toUpperCase()}-SHARED
						{/if}

						{option.label}
					</option>
				{/each}
			</select>

			<span>--spec-type</span>

			<span>{specType}</span>
		{/if}
	</div>

	<button
		aria-label="Copy command"
		class="absolute top-1/2 right-2 -translate-y-1/2 cursor-pointer rounded-md p-1.5 text-muted-foreground/70 transition-colors hover:bg-primary/10 hover:text-foreground"
		onclick={copy}
		type="button"
	>
		{#if copied}
			<Check class="h-3.5 w-3.5 text-green-500" />
		{:else}
			<Copy class="h-3.5 w-3.5" />
		{/if}
	</button>
</div>
