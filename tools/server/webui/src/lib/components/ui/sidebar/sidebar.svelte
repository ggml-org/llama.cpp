<script lang="ts">
	import { cn, type WithElementRef } from '$lib/components/ui/utils.js';
	import type { HTMLAttributes } from 'svelte/elements';
	import { useSidebar } from './context.svelte.js';

	let {
		ref = $bindable(null),
		side = 'left',
		variant = 'sidebar',
		collapsible = 'offcanvas',
		class: className,
		children,
		...restProps
	}: WithElementRef<HTMLAttributes<HTMLDivElement>> & {
		side?: 'left' | 'right';
		variant?: 'sidebar' | 'floating' | 'inset';
		collapsible?: 'offcanvas' | 'icon' | 'none';
	} = $props();

	const sidebar = useSidebar();
</script>

{#if collapsible === 'none'}
	<div
		class={cn(
			'flex h-full w-(--sidebar-width) flex-col bg-sidebar text-sidebar-foreground',
			className
		)}
		bind:this={ref}
		{...restProps}
	>
		{@render children?.()}
	</div>
{:else}
	<div
		bind:this={ref}
		class="group peer block text-sidebar-foreground"
		data-state={sidebar.state}
		data-collapsible={sidebar.state === 'collapsed' ? collapsible : ''}
		data-variant={variant}
		data-side={side}
		data-slot="sidebar"
	>
		<!-- This is what handles the sidebar gap on desktop -->
		<div
			data-slot="sidebar-gap"
			class={cn(
				'relative bg-transparent transition-[width] duration-200 ease-linear',
				'w-0',
				variant === 'floating'
					? 'md:w-[calc(var(--sidebar-width)+0.75rem)]'
					: 'md:w-(--sidebar-width)',
				'md:group-data-[collapsible=offcanvas]:w-0',
				'group-data-[side=right]:rotate-180',
				variant === 'floating' || variant === 'inset'
					? 'group-data-[collapsible=icon]:w-[calc(var(--sidebar-width-icon)+(--spacing(4))+2px)]'
					: 'group-data-[collapsible=icon]:w-(--sidebar-width-icon)'
			)}
		></div>

		<div
			data-slot="sidebar-container"
			class={cn(
				'fixed inset-y-0 z-[900] flex w-[calc(100dvw-1.5rem)] duration-200 ease-linear md:z-0 md:w-(--sidebar-width)',
				variant === 'floating'
					? [
							'transition-[left,right,width,opacity]',
							side === 'left'
								? 'left-3 group-data-[collapsible=offcanvas]:left-[calc(var(--sidebar-width)*-0.775)] group-data-[collapsible=offcanvas]:opacity-0'
								: 'right-3 group-data-[collapsible=offcanvas]:right-[calc(var(--sidebar-width)*-0.775)] group-data-[collapsible=offcanvas]:opacity-0',
							'my-3 overflow-hidden rounded-3xl border border-sidebar-border shadow-md'
						]
					: [
							'h-svh transition-[left,right,width]',
							side === 'left'
								? 'left-0 group-data-[collapsible=offcanvas]:left-[calc(var(--sidebar-width)*-1)]'
								: 'right-0 group-data-[collapsible=offcanvas]:right-[calc(var(--sidebar-width)*-1)]'
						],
				// Adjust the padding for inset variant.
				variant === 'inset'
					? 'p-2 group-data-[collapsible=icon]:w-[calc(var(--sidebar-width-icon)+(--spacing(4))+2px)]'
					: variant === 'floating'
						? 'group-data-[collapsible=icon]:w-[calc(var(--sidebar-width-icon)+(--spacing(4))+2px)]'
						: 'group-data-[collapsible=icon]:w-(--sidebar-width-icon)',
				className
			)}
			style={variant === 'floating' ? 'height: calc(100dvh - 1.5rem);' : undefined}
			{...restProps}
		>
			<div
				data-sidebar="sidebar"
				data-slot="sidebar-inner"
				class="flex h-full w-full flex-col bg-sidebar"
			>
				{@render children?.()}
			</div>
		</div>
	</div>
{/if}
