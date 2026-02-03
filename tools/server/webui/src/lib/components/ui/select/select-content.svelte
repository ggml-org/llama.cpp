<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { Select as SelectPrimitive } from 'bits-ui';
	import SelectScrollUpButton from './select-scroll-up-button.svelte';
	import SelectScrollDownButton from './select-scroll-down-button.svelte';
	import { cn, type WithoutChild } from '$lib/components/ui/utils.js';

	let {
		ref = $bindable(null),
		class: className,
		sideOffset = 4,
		portalProps,
		children,
		...restProps
	}: WithoutChild<SelectPrimitive.ContentProps> & {
		portalProps?: SelectPrimitive.PortalProps;
	} = $props();

	let cleanupInternalListeners: (() => void) | undefined;

	onMount(() => {
		const listenerOptions: AddEventListenerOptions = { passive: false };

		const blockOutsideWheel = (event: WheelEvent) => {
			if (!ref) {
				return;
			}

			const target = event.target as Node | null;

			if (!target || !ref.contains(target)) {
				event.preventDefault();
				event.stopPropagation();
			}
		};

		const blockOutsideTouchMove = (event: TouchEvent) => {
			if (!ref) {
				return;
			}

			const target = event.target as Node | null;

			if (!target || !ref.contains(target)) {
				event.preventDefault();
				event.stopPropagation();
			}
		};

		document.addEventListener('wheel', blockOutsideWheel, listenerOptions);
		document.addEventListener('touchmove', blockOutsideTouchMove, listenerOptions);

		return () => {
			document.removeEventListener('wheel', blockOutsideWheel, listenerOptions);
			document.removeEventListener('touchmove', blockOutsideTouchMove, listenerOptions);
		};
	});

	$effect(() => {
		const element = ref;

		cleanupInternalListeners?.();

		if (!element) {
			return;
		}

		const stopWheelPropagation = (event: WheelEvent) => {
			event.stopPropagation();
		};

		const stopTouchPropagation = (event: TouchEvent) => {
			event.stopPropagation();
		};

		element.addEventListener('wheel', stopWheelPropagation);
		element.addEventListener('touchmove', stopTouchPropagation);

		cleanupInternalListeners = () => {
			element.removeEventListener('wheel', stopWheelPropagation);
			element.removeEventListener('touchmove', stopTouchPropagation);
		};
	});

	onDestroy(() => {
		cleanupInternalListeners?.();
	});
</script>

<SelectPrimitive.Portal {...portalProps}>
	<SelectPrimitive.Content
		bind:ref
		{sideOffset}
		data-slot="select-content"
		class={cn(
			'relative z-[var(--layer-popover,1000000)] max-h-(--bits-select-content-available-height) min-w-[8rem] origin-(--bits-select-content-transform-origin) overflow-x-hidden overflow-y-auto rounded-md border bg-popover text-popover-foreground shadow-md data-[side=bottom]:translate-y-1 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:-translate-x-1 data-[side=left]:slide-in-from-right-2 data-[side=right]:translate-x-1 data-[side=right]:slide-in-from-left-2 data-[side=top]:-translate-y-1 data-[side=top]:slide-in-from-bottom-2 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95',
			className
		)}
		{...restProps}
	>
		<SelectScrollUpButton />
		<SelectPrimitive.Viewport
			class={cn(
				'h-(--bits-select-anchor-height) w-full min-w-(--bits-select-anchor-width) scroll-my-1 p-1'
			)}
		>
			{@render children?.()}
		</SelectPrimitive.Viewport>
		<SelectScrollDownButton />
	</SelectPrimitive.Content>
</SelectPrimitive.Portal>
