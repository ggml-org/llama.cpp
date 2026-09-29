<script lang="ts">
	import {
		ActionIconCopyToClipboard,
		BadgesModality,
		CollapsibleSection
	} from '$lib/components/app';
	import { Badge } from '$lib/components/ui/badge';
	import { HuggingFaceService } from '$lib/services';
	import { modelsStore } from '$lib/stores';
	import type { ApiLlamaCppServerProps } from '$lib/types/api';
	import type { HfModelDetailInfo } from '$lib/types/huggingface';
	import type { ModelOption } from '$lib/types/models';
	import { formatFileSize, formatNumber, formatParameters } from '$lib/utils/formatters';

	interface Props {
		/** Hub details, when the Hub is enabled and the server reports none. */
		hub?: HfModelDetailInfo | null;
		option: ModelOption;
		serverProps: ApiLlamaCppServerProps | null | undefined;
	}

	let { hub = null, option, serverProps }: Props = $props();

	let meta = $derived(option.meta);
	let gguf = $derived(hub?.gguf ?? null);
	let quant = $derived(option.parsedId?.quantization ?? null);
	// a GGUF carries the count in its metadata, a transformers repo in its SafeTensors index
	let hubParameters = $derived(HuggingFaceService.parameterCount(hub));
	let draftSidecars = $derived(option.draftSidecars ?? []);

	let modalities = $derived.by(() => {
		void modelsStore.props.cacheVersion;

		return modelsStore.props.getModelModalitiesArray(option.id);
	});

	let rows = $derived([
		{
			isCopyable: true,
			isMono: true,
			label: 'File Path',
			value: serverProps?.model_path ?? null
		},
		{
			label: 'Context Size',
			// the server reports the context it runs with once the model is loaded; until
			// then the listing, or the Hub, says what the model can take
			value: serverProps
				? `${formatNumber(serverProps.default_generation_settings?.n_ctx ?? 0)} tokens`
				: option.contextLength
					? `${formatParameters(option.contextLength)} tokens`
					: gguf?.context_length
						? `${formatNumber(gguf.context_length)} tokens`
						: null
		},
		{ label: 'Model Size', value: meta?.size ? formatFileSize(meta.size) : null },
		{
			label: 'Parameters',
			value: meta?.n_params
				? formatParameters(meta.n_params)
				: (option.parsedId?.params ?? (hubParameters ? formatParameters(hubParameters) : null))
		},
		{ label: 'Embedding Size', value: meta?.n_embd ? formatNumber(meta.n_embd) : null },
		{
			label: 'Vocabulary Size',
			value: meta?.n_vocab ? `${formatNumber(meta.n_vocab)} tokens` : null
		},
		{ isBadge: true, label: 'Quantization', value: quant },
		{
			isBadge: true,
			label: 'Architecture',
			value: (meta?.architecture as string) ?? gguf?.architecture ?? null
		},
		{
			isBadge: true,
			label: 'Draft sidecars',
			value:
				draftSidecars.length > 0
					? draftSidecars.map((badge) => badge.kind.toUpperCase()).join(', ')
					: null
		},
		{
			isCopyable: true,
			label: 'Draft repo',
			value: draftSidecars[0]?.repo ?? null
		},
		{
			isBadge: true,
			label: 'Draft quant',
			value: draftSidecars[0]?.quant ?? null
		},
		{
			label: 'Parallel Slots',
			value: serverProps?.total_slots != null ? String(serverProps.total_slots) : null
		},
		{ isMono: true, label: 'Build Info', value: serverProps?.build_info ?? null }
	] satisfies Array<{
		isBadge?: boolean;
		isCopyable?: boolean;
		isMono?: boolean;
		label: string;
		value: string | null;
	}>);

	// rows without a value say nothing; a loaded model fills the rest in from /props
	let visibleRows = $derived(rows.filter((row) => row.value !== null));

	const sectionTrigger = 'flex w-full cursor-pointer items-center gap-2 py-2 text-left';
</script>

<div class="space-y-0">
	<div class="flex items-center gap-3 border-b border-border/30 py-2.5">
		<span class="text-sm text-muted-foreground">Model</span>

		<span class="ml-auto flex min-w-0 flex-1 items-center gap-2">
			<span class="min-w-0 flex-1 overflow-x-auto font-mono text-xs whitespace-nowrap">
				{option.model}
			</span>

			<ActionIconCopyToClipboard
				ariaLabel="Copy model name to clipboard"
				canCopy
				text={option.model}
			/>
		</span>
	</div>

	{#each visibleRows as row (row.label)}
		<div class="flex items-center gap-3 border-b border-border/30 py-2.5 last:border-b-0">
			<span class="text-sm text-muted-foreground">{row.label}</span>

			<span class="ml-auto flex min-w-0 items-center gap-2 {row.isCopyable ? 'flex-1' : ''}">
				{#if row.value === null}
					<span class="text-muted-foreground">-</span>
				{:else}
					<span
						class={[
							'min-w-0',
							// a copyable value scrolls instead of clipping, the icon stays put
							row.isCopyable ? 'flex-1 overflow-x-auto whitespace-nowrap' : 'truncate',
							row.isBadge ? '' : 'text-sm',
							row.isMono ? 'font-mono text-xs' : ''
						]}
					>
						{#if row.isBadge}
							<Badge class="h-5 px-1.5 text-[10px]" variant="secondary">{row.value}</Badge>
						{:else}
							{row.value}
						{/if}
					</span>

					{#if row.isCopyable}
						<ActionIconCopyToClipboard
							ariaLabel="Copy value to clipboard"
							canCopy
							text={row.value}
						/>
					{/if}
				{/if}
			</span>
		</div>
	{/each}

	{#if visibleRows.length === 0}
		<p class="py-2.5 text-sm text-muted-foreground">This model reports no metadata.</p>
	{/if}

	{#if modalities.length > 0}
		<div class="flex items-center gap-3 border-b border-border/30 py-2.5 last:border-b-0">
			<span class="text-sm text-muted-foreground">Modalities</span>

			<span class="ml-auto flex flex-wrap gap-1"><BadgesModality {modalities} /></span>
		</div>
	{/if}
</div>

{#if !serverProps}
	<p class="pt-2 text-xs text-muted-foreground">
		/props values - file path, slots, build info - appear once the model is loaded. Reading this
		page never loads a model.
	</p>
{/if}

<div class="mt-4">
	<CollapsibleSection triggerClass={sectionTrigger}>
		{#snippet trigger()}
			<span class="text-sm font-medium">Chat Template</span>
		{/snippet}

		<pre
			class="mt-1 max-h-48 overflow-auto rounded-md bg-muted/50 p-2 text-xs whitespace-pre-wrap">{serverProps?.chat_template ??
				gguf?.chat_template ??
				'Not reported by the server.'}</pre>
	</CollapsibleSection>
</div>
