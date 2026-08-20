<script lang="ts">
	import { Loader2, Square, SquarePen, X } from '@lucide/svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { cn } from '$lib/components/ui/utils';

	interface Tab {
		id: string;
		isNewChat: boolean;
		name: string;
	}

	interface Props {
		tab: Tab;
		isActive?: boolean;
		isLoading?: boolean;
		onActivate?: (id: string) => void;
		onClose?: (id: string) => void;
		onStop?: (id: string, event: MouseEvent) => void;
		onAuxClick?: (id: string, event: MouseEvent) => void;
	}

	let {
		isActive = false,
		isLoading = false,
		onActivate,
		onAuxClick,
		onClose,
		onStop,
		tab
	}: Props = $props();

	let contentOpacity = $derived(isActive ? '' : 'opacity-40 group-hover:opacity-100');
</script>

<div
	data-active-tab={isActive ? 'true' : undefined}
	class={cn(
		'flex h-8 max-w-52 min-w-0 shrink-0 items-center gap-1 rounded-lg pr-1 text-sm whitespace-nowrap transition-[background-color,border-color,box-shadow] hover:bg-foreground/10 border backdrop-blur-xl first:ml-2',
		isLoading ? 'pl-1' : 'pl-3',
		isActive
			? 'bg-muted/60 border-border/10 shadow-sm text-accent-foreground hover:bg-primary/15'
			: 'border-transparent hover:bg-primary/10 hover:border-border/10 hover:shadow-sm'
	)}
>
	{#if isLoading}
		<Tooltip.Root>
			<Tooltip.Trigger>
				{#snippet child({ props })}
					<button
						{...props}
						class="stop-button flex h-5 w-5 shrink-0 cursor-pointer items-center justify-center rounded-sm text-muted-foreground transition-colors hover:text-foreground"
						onclick={(e) => onStop?.(tab.id, e)}
						aria-label="Stop generation"
					>
						<Loader2
							class="loading-icon h-3.5 w-3.5 animate-spin transition-opacity {contentOpacity}"
						/>
						<Square
							class="stop-icon hidden h-3 w-3 fill-current text-destructive transition-opacity {contentOpacity}"
						/>
					</button>
				{/snippet}
			</Tooltip.Trigger>

			<Tooltip.Content>
				<p>Stop generation</p>
			</Tooltip.Content>
		</Tooltip.Root>
	{/if}

	<button
		class="flex min-w-0 flex-1 cursor-pointer items-center gap-2"
		onclick={() => onActivate?.(tab.id)}
		onauxclick={(e) => onAuxClick?.(tab.id, e)}
		aria-current={isActive ? 'page' : undefined}
	>
		{#if tab.isNewChat}
			<SquarePen class="h-3.5 w-3.5 shrink-0 transition-opacity {contentOpacity}" />
		{/if}

		<span class="truncate transition-opacity {contentOpacity}">{tab.name}</span>
	</button>

	<Tooltip.Root>
		<Tooltip.Trigger>
			{#snippet child({ props })}
				<button
					{...props}
					class={cn(
						'flex h-5 w-5 shrink-0 cursor-pointer items-center justify-center rounded-sm text-muted-foreground transition-opacity hover:bg-foreground/10 hover:text-foreground'
					)}
					onclick={() => onClose?.(tab.id)}
					aria-label="Close tab"
				>
					<X class="h-3.5 w-3.5" />
				</button>
			{/snippet}
		</Tooltip.Trigger>

		<Tooltip.Content>
			<p>Close tab</p>
		</Tooltip.Content>
	</Tooltip.Root>
</div>

<style>
	.stop-button {
		:global(.stop-icon) {
			display: none;
		}

		:global(.loading-icon) {
			display: block;
		}

		&:is(:hover) {
			:global(.stop-icon) {
				display: block;
			}

			:global(.loading-icon) {
				display: none;
			}
		}
	}
</style>
