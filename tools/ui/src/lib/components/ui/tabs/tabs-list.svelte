<script lang="ts" module>
	import { tv } from '$lib/components/ui/utils';
	import type { VariantProps } from 'tailwind-variants';

	// default: the segmented track, 2px around the triggers, as tall as a default control
	export const tabsListVariants = tv({
		base: 'group/tabs-list relative inline-flex w-fit items-center justify-center text-(--tabs-text) group-data-[orientation=horizontal]/tabs:h-control-md group-data-[orientation=vertical]/tabs:h-fit group-data-[orientation=vertical]/tabs:flex-col',
		defaultVariants: {
			variant: 'default'
		},
		variants: {
			variant: {
				default: 'gap-0.5 rounded-panel bg-(--tabs-track) p-0.5 shadow-(--tabs-track-shadow)',
				line: 'gap-1 bg-transparent'
			}
		}
	});

	export type TabsListVariant = VariantProps<typeof tabsListVariants>['variant'];

	/** Gap between a trigger and the line variant's underline, and the underline's thickness. */
	const LINE_OFFSET = 2;
	const LINE_WIDTH = 2;

	// a plain ease-out with no overshoot: a bounce would carry the first and last tab past the track
	const SLIDE_EASE = 'cubic-bezier(0.23, 1, 0.32, 1)';
	const SLIDE_MS = 300;
	const SIZE_MS = 250;

	/** Stretch along the move, 2% to 5% with the distance, and half that as squash across it. */
	function stretchFor(distance: number): number {
		return Math.min(0.05, 0.02 + distance / 4000);
	}
</script>

<script lang="ts">
	import { cn } from '$lib/components/ui/utils.js';
	import { Tabs as TabsPrimitive } from 'bits-ui';
	import { untrack } from 'svelte';
	import { prefersReducedMotion } from 'svelte/motion';

	let {
		children,
		class: className,
		ref = $bindable(null),
		variant = 'default',
		...restProps
	}: TabsPrimitive.ListProps & {
		variant?: TabsListVariant;
	} = $props();

	// the active tab's box inside the list; the indicator slides to it
	let box = $state<{ height: number; width: number; x: number; y: number } | null>(null);
	// off until the indicator has been placed once, so it does not fly in from the corner;
	// a plain flag, since the effect below must not depend on it
	let isAnimated = $state(false);
	let isPlaced = false;
	let indicatorEl: HTMLSpanElement | null = $state(null);
	// the last measured box, kept outside state so measuring never re-runs the effect
	let lastBox: { x: number; y: number } | null = null;

	let indicator = $derived.by(() => {
		if (!box) return null;

		if (variant !== 'line') return box;

		const vertical = ref?.dataset.orientation === 'vertical';

		return vertical
			? { height: box.height, width: LINE_WIDTH, x: box.x + box.width + LINE_OFFSET, y: box.y }
			: { height: LINE_WIDTH, width: box.width, x: box.x, y: box.y + box.height + LINE_OFFSET };
	});

	function measure() {
		const active = ref?.querySelector<HTMLElement>(
			'[data-slot="tabs-trigger"][data-state="active"]'
		);

		// a transition switched on in the same update as a move still runs for that move
		if (isPlaced && active) isAnimated = true;

		isPlaced = !!active;

		box = active
			? {
					height: active.offsetHeight,
					width: active.offsetWidth,
					x: active.offsetLeft,
					y: active.offsetTop
				}
			: null;

		if (box && lastBox && isAnimated) stretch(box.x - lastBox.x, box.y - lastBox.y);

		lastBox = box && { x: box.x, y: box.y };
	}

	/** A barely visible stretch along the move that settles as the slide lands. */
	function stretch(dx: number, dy: number) {
		const distance = Math.hypot(dx, dy);

		if (!indicatorEl || distance < 1 || prefersReducedMotion.current) return;

		const amount = stretchFor(distance);
		const along = 1 + amount;
		// the underline is too thin to squash; only the chip narrows across the move
		const across = variant === 'line' ? 1 : 1 - amount / 2;
		const peak = Math.abs(dx) >= Math.abs(dy) ? `${along} ${across}` : `${across} ${along}`;

		indicatorEl.animate([{ scale: '1 1' }, { offset: 0.35, scale: peak }, { scale: '1 1' }], {
			duration: SLIDE_MS,
			easing: 'ease-out'
		});
	}

	$effect(() => {
		const list = ref;

		if (!list) return;

		// measure reads state of its own; only a new list element should re-run this
		untrack(measure);

		// a new active tab flips data-state; labels and layout changes resize the triggers
		const mutations = new MutationObserver(measure);

		mutations.observe(list, { attributeFilter: ['data-state'], childList: true, subtree: true });

		const resizes = new ResizeObserver(measure);

		resizes.observe(list);
		for (const trigger of list.querySelectorAll('[data-slot="tabs-trigger"]')) {
			resizes.observe(trigger);
		}

		return () => {
			mutations.disconnect();
			resizes.disconnect();
		};
	});
</script>

<TabsPrimitive.List
	bind:ref
	class={cn(tabsListVariants({ variant }), className)}
	data-slot="tabs-list"
	data-variant={variant}
	{...restProps}
>
	<span
		bind:this={indicatorEl}
		style:height={indicator ? `${indicator.height}px` : undefined}
		style:transition={isAnimated && !prefersReducedMotion.current
			? `translate ${SLIDE_MS}ms ${SLIDE_EASE}, width ${SIZE_MS}ms ${SLIDE_EASE}, height ${SIZE_MS}ms ${SLIDE_EASE}`
			: undefined}
		style:translate={indicator ? `${indicator.x}px ${indicator.y}px` : undefined}
		style:width={indicator ? `${indicator.width}px` : undefined}
		aria-hidden="true"
		class={cn(
			'pointer-events-none absolute top-0 left-0',
			variant === 'line'
				? 'rounded-full bg-(--tabs-line-active)'
				: 'rounded-control bg-(--tabs-background-active) shadow-(--tabs-shadow-active)',
			!indicator && 'hidden'
		)}
		data-slot="tabs-indicator"
	></span>

	{@render children?.()}
</TabsPrimitive.List>
