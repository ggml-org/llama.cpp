<script lang="ts">
	import ToolCallBlock from './ToolCallBlock.svelte';
	import { XCircle } from '@lucide/svelte';
	import { toolsStore } from '$lib/stores';
	import type { AgenticSection } from '$lib/types';
	import { abbreviateHome } from '$lib/utils';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		onToggle?: () => void;
	}

	let { isStreaming, onToggle, open, section }: Props = $props();

	type GetInfoMeta = {
		os?: string;
		cwd?: string;
		errorMessage?: string;
	};

	function parseGetInfoMeta(toolResultString: string | undefined): GetInfoMeta {
		if (!toolResultString) return {};

		try {
			const parsed: unknown = JSON.parse(toolResultString);

			if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
				const obj = parsed as Record<string, unknown>;

				if (typeof obj.error === 'string') return { errorMessage: obj.error };

				return {
					cwd: typeof obj.cwd === 'string' ? obj.cwd : undefined,
					os: typeof obj.os === 'string' ? obj.os : undefined
				};
			}
		} catch {
			// not JSON - nothing to show
		}

		return {};
	}

	const infoMeta = $derived(parseGetInfoMeta(section.toolResult));
	const home = $derived(toolsStore.serverHome);
	const cwdDisplay = $derived(abbreviateHome(infoMeta.cwd ?? '', home));
</script>

<ToolCallBlock
	{isStreaming}
	meta={infoMeta}
	{onToggle}
	{open}
	{section}
	spinIconWhenActive
	title="Runtime info"
>
	{#snippet children(meta, _ctx)}
		{#if meta?.errorMessage}
			<div
				class="flex items-start gap-2 rounded bg-red-500/10 p-2 text-xs text-red-600 italic dark:text-red-400"
			>
				<XCircle class="mt-0.5 h-3 w-3 shrink-0" />

				<span>{meta.errorMessage}</span>
			</div>
		{:else if infoMeta.os || infoMeta.cwd}
			<table class="w-full table-fixed border-collapse text-sm">
				<colgroup>
					<col class="w-12" />

					<col />
				</colgroup>

				<tbody class="divide-y divide-border/50">
					{#if infoMeta.os}
						<tr>
							<th
								class="py-1 pr-3 text-left align-baseline text-[11px] font-medium tracking-wide text-muted-foreground/60 uppercase"
								scope="row"
							>
								os
							</th>

							<td class="py-1 align-baseline">
								<div class="min-w-0 overflow-x-auto font-mono text-foreground/90">
									{infoMeta.os}
								</div>
							</td>
						</tr>
					{/if}

					{#if infoMeta.cwd}
						<tr>
							<th
								class="py-1 pr-3 text-left align-baseline text-[11px] font-medium tracking-wide text-muted-foreground/60 uppercase"
								scope="row"
							>
								cwd
							</th>

							<td class="py-1 align-baseline">
								<div
									class="min-w-0 overflow-x-auto font-mono text-foreground/90"
									title={infoMeta.cwd}
								>
									{cwdDisplay}
								</div>
							</td>
						</tr>
					{/if}
				</tbody>
			</table>
		{:else if section.toolResult}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
				{section.toolResult}
			</div>
		{:else}
			<div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
				Waiting for runtime info...
			</div>
		{/if}
	{/snippet}
</ToolCallBlock>
