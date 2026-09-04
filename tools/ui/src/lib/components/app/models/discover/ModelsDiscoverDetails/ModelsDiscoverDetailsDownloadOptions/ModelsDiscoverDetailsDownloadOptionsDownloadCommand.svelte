<script lang="ts">
	import { quantBitDepth } from './download-options.utils';
	import { Check, Copy, Plus, X } from '@lucide/svelte';
	import * as Select from '$lib/components/ui/select';
	import { DEFAULT_BASE_BIT_DEPTH, type ModelSidecar,SPEC_TYPE } from '$lib/constants';
	import { HuggingFaceService } from '$lib/services';
	import type { QuantOption } from '$lib/types';
	import { copyToClipboard } from '$lib/utils';

	interface Props {
		modelId: string;
		/** Non-draft quants of the repo, in bit-depth row order. */
		mainOptions: QuantOption[];
		/** Draft sidecar files with their sidecar badge; empty when the repo ships none. */
		draftOptions: (QuantOption & { badge: ModelSidecar | null })[];
	}

	let { draftOptions, mainOptions, modelId }: Props = $props();

	// Command picks, owned here: nothing two-way binds them to the quant chips.
	let basePick = $state<string | null>(null);
	let draftPick = $state<string | null>(null);
	let draftTypePick = $state<ModelSidecar | null>(null);
	let withDraft = $state(false);

	function bitDepthOf(path: string): number {
		return quantBitDepth(HuggingFaceService.extractQuantMeta(path)?.quant ?? null);
	}

	/**
	 * Base file the command points at: the user's pick while it still exists in
	 * the options, else the 4-bit file, else the lowest bit depth available. A
	 * stale pick (the details pane switched models) falls back on its own.
	 */
	let baseOption = $derived.by(() => {
		const picked = mainOptions.find((option) => option.path === basePick);

		if (picked) return picked;

		const preferred = mainOptions.find(
			(option) => bitDepthOf(option.path) === DEFAULT_BASE_BIT_DEPTH
		);

		if (preferred) return preferred;

		const ranked = [...mainOptions].sort((a, b) => bitDepthOf(a.path) - bitDepthOf(b.path));

		return ranked[0] ?? null;
	});

	/** Draft sidecar types the repo ships, in option order. */
	let specTypes = $derived(
		draftOptions
			.map((option) => option.badge)
			.filter((badge): badge is ModelSidecar => badge !== null)
			.filter((badge, index, all) => all.indexOf(badge) === index)
	);

	/**
	 * Draft type the --spec-type select points at: the user's pick while it
	 * still exists, else the first type the repo ships.
	 */
	let draftType = $derived(
		draftTypePick && specTypes.includes(draftTypePick) ? draftTypePick : (specTypes[0] ?? null)
	);

	/** Draft files of the picked type; the quant select only offers these. */
	let typeDraftOptions = $derived(draftOptions.filter((option) => option.badge === draftType));

	/** Draft file the -hfd tag points at: the user's pick, else the first of the type. */
	let draftOption = $derived(
		withDraft
			? (typeDraftOptions.find((option) => option.path === draftPick) ??
					typeDraftOptions[0] ??
					null)
			: null
	);

	/** Quant of the file the `-hf` tag points at; null when the file carries no quant. */
	let mainQuant = $derived(
		baseOption ? (HuggingFaceService.extractQuantMeta(baseOption.path)?.quant ?? null) : null
	);

	/** Quant of the file the `-hfd` tag points at. */
	let draftQuant = $derived(
		draftOption ? (HuggingFaceService.extractQuantMeta(draftOption.path)?.quant ?? null) : null
	);

	/** `--spec-type` value; null when no draft type resolved. */
	let specType = $derived(draftType ? SPEC_TYPE[draftType] : null);

	/** The llama serve command, composed from the inline picks. */
	let command = $derived.by(() => {
		const parts = ['llama', 'serve', '-hf', mainQuant ? `${modelId}:${mainQuant}` : modelId];

		if (draftOption && draftQuant) {
			parts.push('-hfd', `${modelId}:${draftQuant}`, '--spec-type', specType ?? '');
		}

		return parts.join(' ');
	});

	let copied = $state(false);

	async function copy() {
		await copyToClipboard(command);
		copied = true;
		setTimeout(() => (copied = false), 1500);
	}
</script>

<!-- <div aria-hidden="true" class="flex items-center gap-3 mt-2 mb-4">
	<span class="h-px flex-1 bg-border/50"></span>

	<span class="text-xs whitespace-nowrap text-muted-foreground"> or run in your terminal </span>

	<span class="h-px flex-1 bg-border/50"></span>
</div> -->

<div
	class="relative flex items-center gap-2 overflow-hidden rounded-lg border border-border/40 bg-background py-2.5 pl-4 pr-10 shadow-xs dark:border-border/35 dark:bg-background/50"
>
	<!-- Single line: long commands scroll horizontally instead of wrapping. -->
	<div
		class="flex min-w-0 flex-1 items-center gap-x-2 overflow-x-auto py-0.5 font-mono text-xs whitespace-nowrap text-foreground/90"
	>
		<span class="shrink-0">llama</span>

		<span class="shrink-0">serve</span>

		<span class="shrink-0">-hf</span>

		<span class="shrink-0">{modelId}{mainQuant ? ':' : ''}</span>

		<!-- Base quant: always part of the command, the 4-bit file by default. -->
		{#if baseOption}
			<Select.Root onValueChange={(v) => v && (basePick = v)} type="single" value={baseOption.path}>
				<Select.Trigger
					aria-label="Base model quantization"
					class="-ml-2 border-primary/15 bg-primary/[0.07] font-mono text-foreground hover:bg-primary/15 focus-visible:border-primary/40 focus-visible:ring-0"
					size="xs"
				>
					{baseOption.label}
				</Select.Trigger>

				<Select.Content class="font-mono text-xs">
					{#each mainOptions as option (option.path)}
						<Select.Item class="text-xs" label={option.label} value={option.path}>
							{option.label}
						</Select.Item>
					{/each}
				</Select.Content>
			</Select.Root>
		{/if}

		<!-- Draft add: a tiny dashed affordance right of the base part; gone once added. -->
		{#if draftOptions.length && !withDraft}
			<button
				aria-label="Add draft model"
				class="mx-1 inline-flex h-5 shrink-0 cursor-pointer items-center gap-1 rounded-md border border-dashed border-border/60 px-1.5 text-[10px] text-muted-foreground transition-colors hover:border-border hover:text-foreground"
				onclick={() => (withDraft = true)}
				type="button"
			>
				<Plus class="h-3 w-3" />

				add draft model
			</button>
		{/if}

		<!-- Draft segment: quant and spec type of the picked draft flavour. The X
				 at the end drops the whole segment (the add button is gone once added);
				 it only appears while hovering the segment, or directly on touch -->
		{#if draftOption}
			<span class="group/draft inline-flex shrink-0 items-center gap-x-2">
				<span>-hfd</span>

				<span class="shrink-0">{modelId}{draftQuant ? ':' : ''}</span>

				<Select.Root
					onValueChange={(v) => v && (draftPick = v)}
					type="single"
					value={draftOption.path}
				>
					<Select.Trigger
						aria-label="Draft model quantization"
						class="-ml-2 border-primary/15 bg-primary/[0.07] font-mono text-foreground hover:bg-primary/15 focus-visible:border-primary/40 focus-visible:ring-0"
						size="xs"
					>
						{draftOption.label}
					</Select.Trigger>

					<Select.Content class="font-mono text-xs">
						{#each typeDraftOptions as option (option.path)}
							<Select.Item class="text-xs" label={option.label} value={option.path}>
								{option.label}
							</Select.Item>
						{/each}
					</Select.Content>
				</Select.Root>

				{#if draftType}
					<span>--spec-type</span>

					<!-- the select only earns its chrome when there is a real choice to make -->
					{#if specTypes.length > 1}
						<Select.Root
							onValueChange={(v) => v && (draftTypePick = v as ModelSidecar)}
							type="single"
							value={draftType}
						>
							<Select.Trigger
								aria-label="Draft type"
								class="border-primary/15 bg-primary/[0.07] font-mono text-foreground hover:bg-primary/15 focus-visible:border-primary/40 focus-visible:ring-0"
								size="xs"
							>
								{SPEC_TYPE[draftType]}
							</Select.Trigger>

							<Select.Content class="font-mono text-xs">
								{#each specTypes as type (type)}
									<Select.Item class="text-xs" label={SPEC_TYPE[type]} value={type}>
										{SPEC_TYPE[type]}
									</Select.Item>
								{/each}
							</Select.Content>
						</Select.Root>
					{:else}
						<span>{SPEC_TYPE[draftType]}</span>
					{/if}
				{/if}

				<button
					aria-label="Remove draft model"
					class="shrink-0 cursor-pointer text-muted-foreground/60 opacity-0 transition-[opacity,color] duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] hover:text-destructive group-hover/draft:opacity-100 [@media(pointer:coarse)]:opacity-100"
					onclick={() => (withDraft = false)}
					type="button"
				>
					<X class="h-3 w-3" />
				</button>
			</span>
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
