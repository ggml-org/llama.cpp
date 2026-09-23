<script lang="ts">
	import { isLocalOption, modelQuantLabel, modelSizeLabel, resolveModelSize } from '../utils';
	import { CollapsibleSection } from '$lib/components/app';
	import { Badge } from '$lib/components/ui/badge';
	import type { ApiLlamaCppServerProps } from '$lib/types/api';
	import type { ModelOption } from '$lib/types/models';
	import { formatParameters } from '$lib/utils/formatters';

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

	let rows = $derived([
		{ label: 'Model', value: option.model },
		{ label: 'File Path', value: serverProps?.model_path ?? null },
		{
			label: 'Context Size',
			value: serverProps
				? `${formatParameters(serverProps.default_generation_settings.n_ctx)} tokens`
				: null
		},
		{
			label: 'Training Context',
			value: option.contextLength ? `${formatParameters(option.contextLength)} tokens` : null
		},
		{ label: 'Model Size', value: resolvedSize ?? size },
		{ label: 'Parameters', value: option.parsedId?.params ?? null },
		{ isBadge: true, label: 'Quantization', value: quant },
		{ isBadge: true, label: 'Architecture', value: (option.meta?.architecture as string) ?? null },
		{ label: 'Format', value: isLocalOption(option) ? 'GGUF' : null },
		{ label: 'Parallel Slots', value: serverProps ? String(serverProps.total_slots) : null }
	] satisfies Array<{ isBadge?: boolean; label: string; value: string | null }>);

	const sectionTrigger = 'flex w-full cursor-pointer items-center gap-2 py-2 text-left';
</script>

<div class="space-y-0">
	{#each rows as row (row.label)}
		<div class="flex items-center gap-3 border-b border-border/30 py-2.5 last:border-b-0">
			<span class="text-sm text-muted-foreground">{row.label}</span>

			<span class="ml-auto min-w-0 truncate text-sm">
				{#if row.value === null}
					<span class="text-muted-foreground">—</span>
				{:else if row.isBadge}
					<Badge class="h-5 px-1.5 text-[10px]" variant="secondary">{row.value}</Badge>
				{:else}
					{row.value}
				{/if}
			</span>
		</div>
	{/each}
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
