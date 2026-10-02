<script lang="ts">
	import { Loader2, Square, SquarePen, X } from '@lucide/svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { cn } from '$lib/components/ui/utils';
	import { ICON_CLASS_SM, ICON_CLASS_XS, ROUTES, UI_DATA_ATTRS } from '$lib/constants';
	import { RouterService } from '$lib/services/router.service';

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

	let contentOpacity = $derived(isActive ? '' : 'opacity-45 group-hover:opacity-75');

	let href = $derived(tab.isNewChat ? ROUTES.START : RouterService.chat(tab.id));

	function handleActivate(event: MouseEvent) {
		// let cmd/ctrl/middle-click fall through so the browser keeps its own
		// behavior (open in a new window); route the plain click ourselves so the
		// new-chat sentinel and history behave exactly like programmatic nav
		if (event.metaKey || event.ctrlKey || event.button === 1) return;

		event.preventDefault();
		onActivate?.(tab.id);
	}

	// stop/close sit on top of the tab link; swallow their clicks so they do
	// not also navigate
	function handleActionClick(event: MouseEvent, action: () => void) {
		event.preventDefault();
		event.stopPropagation();
		action();
	}
</script>

<!-- the tab link covers the whole item; stop/close sit on top as siblings so
     interactive elements are never nested inside the anchor -->
<div
	{...{ [UI_DATA_ATTRS.ACTIVE_TAB]: isActive ? 'true' : undefined }}
	class={cn(
		'relative flex h-8 max-w-52 min-w-0 shrink-0 items-center gap-1 rounded-lg pr-1 text-sm whitespace-nowrap border backdrop-blur-xl first:ml-2',
		isLoading ? 'pl-1' : 'pl-3',
		isActive
			? 'bg-muted/60 border-border/10 shadow-sm text-accent-foreground hover:bg-primary/15'
			: 'border-transparent hover:bg-primary/10 hover:border-border/10 hover:shadow-sm'
	)}
>
	<a
		aria-current={isActive ? 'page' : undefined}
		aria-label={tab.name}
		class="absolute inset-0 z-0 rounded-lg"
		{href}
		onauxclick={(e) => onAuxClick?.(tab.id, e)}
		onclick={handleActivate}
	></a>

	{#if isLoading}
		<Tooltip.Root>
			<Tooltip.Trigger>
				{#snippet child({ props })}
					<button
						{...props}
						aria-label="Stop generation"
						class="stop-button relative z-10 flex h-5 w-5 shrink-0 cursor-pointer items-center justify-center rounded-sm text-muted-foreground transition-colors hover:text-foreground"
						onclick={(e) => handleActionClick(e, () => onStop?.(tab.id, e))}
					>
						<Loader2
							class="loading-icon {ICON_CLASS_SM} animate-spin transition-opacity duration-300 {contentOpacity}"
						/>

						<Square
							class="stop-icon hidden {ICON_CLASS_XS} fill-current text-destructive transition-opacity {contentOpacity}"
						/>
					</button>
				{/snippet}
			</Tooltip.Trigger>

			<Tooltip.Content>
				<p>Stop generation</p>
			</Tooltip.Content>
		</Tooltip.Root>
	{/if}

	{#if tab.isNewChat}
		<SquarePen
			class="pointer-events-none {ICON_CLASS_SM} shrink-0 transition-opacity {contentOpacity}"
		/>
	{/if}

	<span class="pointer-events-none truncate transition-opacity {contentOpacity}">{tab.name}</span>

	<Tooltip.Root>
		<Tooltip.Trigger>
			{#snippet child({ props })}
				<button
					{...props}
					aria-label="Close tab"
					class={cn(
						'relative z-10 flex h-5 w-5 shrink-0 cursor-pointer items-center justify-center rounded-sm text-muted-foreground transition-opacity hover:bg-foreground/10 hover:text-foreground',
						contentOpacity
					)}
					onclick={(e) => handleActionClick(e, () => onClose?.(tab.id))}
				>
					<X class={ICON_CLASS_SM} />
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
