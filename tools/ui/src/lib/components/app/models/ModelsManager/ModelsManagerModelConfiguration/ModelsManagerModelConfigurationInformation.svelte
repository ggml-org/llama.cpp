<script lang="ts">
	import { modelDraftsFor, modelQuantLabel, modelSizeLabel, resolveModelSize } from '../utils';
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
	import { getBackend } from '$lib/utils/api-base';
	import { getBackendCapabilities } from '$lib/utils/backend';
	import { formatFileSize, formatNumber, formatParameters } from '$lib/utils/formatters';

	interface Props {
		/** Draft the load settings name, when one is set. */
		draftSetting?: string | null;
		/** Hugging Face details, filled in when discovery is on and the server has none. */
		hub?: HfModelDetailInfo | null;
		option: ModelOption;
		serverProps: ApiLlamaCppServerProps | null | undefined;
	}

	let { draftSetting = null, hub = null, option, serverProps }: Props = $props();

	let quant = $derived(modelQuantLabel(option));
	let size = $derived(modelSizeLabel(option));
	// a GGUF carries the count in its metadata, a transformers repo in its SafeTensors index
	let hubParameters = $derived(HuggingFaceService.parameterCount(hub));
	let resolvedSize = $state<string | null>(null);

	// the router rarely reports a size, the repo tree does
	$effect(() => {
		const target = option;

		let cancelled = false;

		resolvedSize = null;

		void resolveModelSize(target)
			.then((value) => {
				if (!cancelled) resolvedSize = value;
			})
			.catch(() => {});

		return () => {
			cancelled = true;
		};
	});

	// the window a model can take, from the listing or from the Hub
	let contextLabel = $derived(
		option.contextLength
			? `${formatParameters(option.contextLength)} tokens`
			: hub?.gguf?.context_length
				? `${formatNumber(hub.gguf.context_length)} tokens`
				: null
	);
	// the draft a load would speculate with, and any other sidecar on disk
	let drafts = $derived(modelDraftsFor(option, draftSetting));
	let activeDraft = $derived(drafts.find((draft) => draft.active) ?? null);
	let idleDrafts = $derived(drafts.filter((draft) => !draft.active));
	let meta = $derived(option.meta);
	// a plain OpenAI-compatible endpoint has no /props or /slots to read
	let reportsServerInfo = $derived(getBackendCapabilities(getBackend(option.backendId)).props);
	// the server reports these once the model is loaded; the Hub knows them anyway
	let gguf = $derived(hub?.gguf ?? null);
	let modalities = $derived.by(() => {
		void modelsStore.props.cacheVersion;

		return modelsStore.props.getModelModalitiesArray(option.id);
	});
	let rows = $derived([
		...(reportsServerInfo
			? [
					{
						isCopyable: true,
						isMono: true,
						label: 'File Path',
						value: serverProps?.model_path ?? null
					}
				]
			: []),
		{
			label: 'Context Size',
			// the server reports the context it runs with once the model is loaded; until then
			// the listing, or the Hub, still says what the model can take
			value: serverProps
				? `${formatNumber(serverProps.default_generation_settings?.n_ctx ?? 0)} tokens`
				: (contextLabel ?? null)
		},
		{
			label: 'Model Size',
			value: meta?.size ? formatFileSize(meta.size) : (resolvedSize ?? size)
		},
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
			value: (option.meta?.architecture as string) ?? gguf?.architecture ?? null
		},
		...(activeDraft
			? [
					{ isBadge: true, label: 'Draft sidecar', value: activeDraft.kind },
					{
						isCopyable: true,
						label: 'Draft model',
						value: activeDraft.model ?? option.model.split(':')[0]
					},
					{ isBadge: true, label: 'Draft quant', value: activeDraft.quant }
				]
			: []),
		{
			isBadge: true,
			label: 'Other sidecars',
			value: idleDrafts.length > 0 ? idleDrafts.map((draft) => draft.kind).join(', ') : null
		},
		...(reportsServerInfo
			? [
					{
						label: 'Parallel Slots',
						value: serverProps?.total_slots != null ? String(serverProps.total_slots) : null
					},
					{ isMono: true, label: 'Build Info', value: serverProps?.build_info ?? null }
				]
			: [])
	] satisfies Array<{
		isBadge?: boolean;
		isCopyable?: boolean;
		isMono?: boolean;
		label: string;
		value: string | null;
	}>);

	// a provider that reports little gets a short list: rows without a value say nothing,
	// and a llama-compat server fills the rest in from /props once the model is loaded
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
					<span class="text-muted-foreground">—</span>
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
		<p class="py-2.5 text-sm text-muted-foreground">This provider reports no model metadata.</p>
	{/if}

	{#if modalities.length > 0}
		<div class="flex items-center gap-3 border-b border-border/30 py-2.5 last:border-b-0">
			<span class="text-sm text-muted-foreground">Modalities</span>

			<span class="ml-auto flex flex-wrap gap-1"><BadgesModality {modalities} /></span>
		</div>
	{/if}
</div>

{#if !reportsServerInfo}
	<p class="pt-2 text-xs text-muted-foreground">
		OpenAI-compatible provider: there is no /props or /slots, so only what the listing and the
		model's repo report is shown.
	</p>
{:else if !serverProps}
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
