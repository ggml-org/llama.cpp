import { untrack } from 'svelte';

/**
 * Scrolls the highlighted row of a picker list into view when the scroll
 * trigger is bumped, without scrolling on mouse hover or result
 * replacement.
 */
export interface UseScrollActiveRowOptions {
	/** Counter bumped by keyboard nav; `undefined` disables the effect. */
	getTrigger: () => number | undefined;
	getContainer: () => HTMLDivElement | null;
	getIndex: () => number;
	getCount: () => number;
	/** Attribute prefix, e.g. 'picker' for `[data-picker-index="0"]`. */
	dataIndex: string;
}

export function useScrollActiveRow(opts: UseScrollActiveRowOptions) {
	let lastTrigger: number | null = null;

	$effect(() => {
		const trigger = opts.getTrigger();
		if (trigger === undefined) return;

		// Skip the initial run on mount: the list opens with the first row
		// already in view, and scrolling here fires before the popover is
		// positioned, which would scroll the whole page to the top.
		if (lastTrigger === null) {
			lastTrigger = trigger;
			return;
		}

		if (trigger === lastTrigger) return;
		lastTrigger = trigger;
		untrack(() => {
			const container = opts.getContainer();
			const index = opts.getIndex();
			if (!container || index < 0 || index >= opts.getCount()) return;
			const row = container.querySelector(
				`[data-${opts.dataIndex}-index="${index}"]`
			) as HTMLElement | null;
			row?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
		});
	});
}

export type UseScrollActiveRowReturn = ReturnType<typeof useScrollActiveRow>;
