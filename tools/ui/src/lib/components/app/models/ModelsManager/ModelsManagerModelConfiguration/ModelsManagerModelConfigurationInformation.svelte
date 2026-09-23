<script lang="ts">
	import { isLocalOption, modelQuantLabel, modelSizeLabel, resolveModelSize } from '../utils';
	import {
		ActionIconCopyToClipboard,
		BadgesModality,
		CollapsibleSection
	} from '$lib/components/app';
	import { Badge } from '$lib/components/ui/badge';
	import { modelsStore } from '$lib/stores';
	import type { ApiLlamaCppServerProps } from '$lib/types/api';
	import type { ModelOption } from '$lib/types/models';
	import { formatFileSize, formatNumber, formatParameters } from '$lib/utils/formatters';

	interface Props {
		option: ModelOption;
		serverProps: ApiLlamaCppServerProps | null | undefined;
	}

	let { option, serverProps }: Props = $props();

	let quant = $derived(modelQuantLabel(option));
	let size = $derived(modelSizeLabel(option));
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

	let meta = $derived(option.meta);
	let modalities = $derived(modelsStore.props.getModelModalitiesArray(option.id));
	let isMissingContext = $derived(!serverProps);
	let rows = $derived([
		{ isCopyable: true, isMono: true, label: 'File Path', value: serverProps?.model_path ?? null },
		{
			label: 'Context Size',
			value: serverProps
				? `${formatNumber(serverProps.default_generation_settings.n_ctx)} tokens`
				: null
		},
		{
			label: 'Training Context',
			value: meta?.n_ctx_train
				? `${formatNumber(meta.n_ctx_train)} tokens`
				: option.contextLength
					? `${formatParameters(option.contextLength)} tokens`
					: null
		},
		{
			label: 'Model Size',
			value: meta?.size ? formatFileSize(meta.size) : (resolvedSize ?? size)
		},
		{
			label: 'Parameters',
			value: meta?.n_params ? formatParameters(meta.n_params) : (option.parsedId?.params ?? null)
		},
		{ label: 'Embedding Size', value: meta?.n_embd ? formatNumber(meta.n_embd) : null },
		{
			label: 'Vocabulary Size',
			value: meta?.n_vocab ? `${formatNumber(meta.n_vocab)} tokens` : null
		},
		{ isCapitalized: true, label: 'Vocabulary Type', value: (meta?.vocab_type as string) ?? null },
		{ isBadge: true, label: 'Quantization', value: quant },
		{ isBadge: true, label: 'Architecture', value: (option.meta?.architecture as string) ?? null },
		{ label: 'Format', value: isLocalOption(option) ? 'GGUF' : null },
		{ label: 'Parallel Slots', value: serverProps ? String(serverProps.total_slots) : null },
		{ isMono: true, label: 'Build Info', value: serverProps?.build_info ?? null }
	] satisfies Array<{
		isBadge?: boolean;
		isCapitalized?: boolean;
		isCopyable?: boolean;
		isMono?: boolean;
		label: string;
		value: string | null;
	}>);

	const sectionTrigger = 'flex w-full cursor-pointer items-center gap-2 py-2 text-left';
</script>

<div class="space-y-0">
	<div class="flex items-center gap-3 border-b border-border/30 py-2.5">
		<span class="text-sm text-muted-foreground">Model</span>

		<span class="ml-auto flex min-w-0 items-center gap-2">
			<span class="min-w-0 truncate font-mono text-xs">{option.model}</span>

			<ActionIconCopyToClipboard
				ariaLabel="Copy model name to clipboard"
				canCopy
				text={option.model}
			/>
		</span>
	</div>

	{#each rows as row (row.label)}
		<div class="flex items-center gap-3 border-b border-border/30 py-2.5 last:border-b-0">
			<span
				class="text-sm text-muted-foreground {row.label === 'Context Size' && isMissingContext
					? 'text-destructive'
					: ''}">{row.label}</span
			>

			<span class="ml-auto flex min-w-0 items-center gap-2">
				{#if row.value === null}
					<span
						class={isMissingContext && row.label === 'Context Size'
							? 'text-destructive'
							: 'text-muted-foreground'}
					>
						{isMissingContext && row.label === 'Context Size' ? 'Not available' : '—'}
					</span>
				{:else}
					<span
						class={[
							'min-w-0 truncate',
							row.isBadge ? '' : 'text-sm',
							row.isCapitalized ? 'capitalize' : '',
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

	{#if modalities.length > 0}
		<div class="flex items-center gap-3 border-b border-border/30 py-2.5 last:border-b-0">
			<span class="text-sm text-muted-foreground">Modalities</span>

			<span class="ml-auto flex flex-wrap gap-1"><BadgesModality {modalities} /></span>
		</div>
	{/if}
</div>

<div class="mt-4">
	<CollapsibleSection triggerClass={sectionTrigger}>
		{#snippet trigger()}
			<span class="text-sm font-medium">Chat Template</span>
		{/snippet}

		<pre
			class="mt-1 max-h-48 overflow-auto rounded-md bg-muted/50 p-2 text-xs whitespace-pre-wrap">{serverProps?.chat_template ??
				'Not reported by the server.'}</pre>
	</CollapsibleSection>

	<CollapsibleSection triggerClass={sectionTrigger}>
		{#snippet trigger()}
			<span class="text-sm font-medium">Source File</span>
		{/snippet}

		<div class="space-y-2 pt-1">
			<p class="text-xs break-all text-muted-foreground">
				{serverProps?.model_path ?? 'Path is reported once the model is loaded.'}
			</p>
		</div>
	</CollapsibleSection>
</div>
