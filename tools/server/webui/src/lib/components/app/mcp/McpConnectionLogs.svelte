<script lang="ts">
	import { ChevronDown, ChevronRight } from '@lucide/svelte';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import { cn } from '$lib/components/ui/utils';
	import type { MCPConnectionLog } from '$lib/types';
	import { formatTime, getMcpLogLevelIcon, getMcpLogLevelClass } from '$lib/utils';

	interface Props {
		logs: MCPConnectionLog[];
		connectionTimeMs?: number;
		defaultExpanded?: boolean;
		class?: string;
	}

	let { logs, connectionTimeMs, defaultExpanded = false, class: className }: Props = $props();

	let isExpanded = $derived(defaultExpanded);

	function formatLogDetails(details: unknown): string {
		if (details == null) {
			return '';
		}

		try {
			return JSON.stringify(details, null, 2);
		} catch {
			return String(details);
		}
	}
</script>

{#if logs.length > 0}
	<Collapsible.Root bind:open={isExpanded} class={className}>
		<div class="space-y-2">
			<Collapsible.Trigger
				class="flex w-full items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
			>
				{#if isExpanded}
					<ChevronDown class="h-3.5 w-3.5" />
				{:else}
					<ChevronRight class="h-3.5 w-3.5" />
				{/if}

				<span>Connection Log ({logs.length})</span>

				{#if connectionTimeMs !== undefined}
					<span class="ml-1">· Connected in {connectionTimeMs}ms</span>
				{/if}
			</Collapsible.Trigger>
		</div>

		<Collapsible.Content class="mt-2">
			<div
				class="max-h-64 space-y-0.5 overflow-y-auto rounded bg-muted/50 p-2 font-mono text-[10px]"
			>
				{#each logs as log (log.timestamp.getTime() + log.message)}
					{@const Icon = getMcpLogLevelIcon(log.level)}

					<div class={cn('flex items-start gap-1.5', getMcpLogLevelClass(log.level))}>
						<span class="shrink-0 text-muted-foreground">
							{formatTime(log.timestamp)}
						</span>

						<Icon class="mt-0.5 h-3 w-3 shrink-0" />

						<span class="break-all">{log.message}</span>
					</div>

					{#if log.details !== undefined}
						<details class="ml-11">
							<summary class="cursor-pointer text-[10px] text-muted-foreground"> details </summary>

							<pre
								class="mt-1 overflow-x-auto rounded bg-background/70 p-2 text-[10px] break-all whitespace-pre-wrap text-foreground/80">
{formatLogDetails(log.details)}</pre>
						</details>
					{/if}
				{/each}
			</div>
		</Collapsible.Content>
	</Collapsible.Root>
{/if}
