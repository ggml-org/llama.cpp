<script lang="ts">
	import { cn, type WithElementRef } from '$lib/components/ui/utils.js';
	import type { HTMLAttributes } from 'svelte/elements';
	import { SIDEBAR_MIN_WIDTH, SIDEBAR_MAX_WIDTH } from './constants.js';
	import { useSidebar } from './context.svelte.js';
	import { remToPx } from '$lib/utils';

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

	function handleResizePointerDown(e: PointerEvent) {
		if (sidebar.isMobile) return;
		e.preventDefault();

		const target = e.currentTarget as HTMLElement;
		target.setPointerCapture(e.pointerId);

		const minPx = remToPx(SIDEBAR_MIN_WIDTH);
		const maxPx = remToPx(SIDEBAR_MAX_WIDTH);

		sidebar.isResizing = true;

		function onPointerMove(ev: PointerEvent) {
			const newWidth = side === 'left' ? ev.clientX : window.innerWidth - ev.clientX;
			const clamped = Math.min(maxPx, Math.max(minPx, newWidth));
			sidebar.sidebarWidth = `${clamped}px`;
		}

		function onPointerUp() {
			sidebar.isResizing = false;
			target.removeEventListener('pointermove', onPointerMove);
			target.removeEventListener('pointerup', onPointerUp);
		}

		target.addEventListener('pointermove', onPointerMove);
		target.addEventListener('pointerup', onPointerUp);
	}
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
				sidebar.isResizing && '!duration-0',
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
				'group-data-[collapsible=offcanvas]:pointer-events-none md:group-data-[collapsible=offcanvas]:pointer-events-auto',
				sidebar.isResizing && '!duration-0',
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
			<!-- Resize handle -->
			{#if side === 'left'}
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
					data-slot="sidebar-resize-handle"
					class="absolute inset-y-0 right-0 z-50 hidden w-1.5 cursor-ew-resize touch-none select-none hover:bg-sidebar-border/50 active:bg-sidebar-border md:block"
					class:bg-sidebar-border={sidebar.isResizing}
					onpointerdown={handleResizePointerDown}
				></div>
			{:else}
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
					data-slot="sidebar-resize-handle"
					class="absolute inset-y-0 left-0 z-50 hidden w-1.5 cursor-ew-resize touch-none select-none hover:bg-sidebar-border/50 active:bg-sidebar-border md:block"
					class:bg-sidebar-border={sidebar.isResizing}
					onpointerdown={handleResizePointerDown}
				></div>
			{/if}
		</div>
	</div>
{/if}
