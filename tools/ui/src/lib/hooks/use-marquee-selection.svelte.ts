/**
 * Reusable selection state-machine: shift+click/shift+drag range select plus
 * rubber-band marquee drag, anchored on the last clicked row. Both the sidebar
 * conversation list and the dialog conversations table share this.
 *
 * The hook mutates the consumer's SvelteSet<string> directly; the consumer
 * owns the source of truth and reads it like any other $state. orderedIds
 * must reflect the current visual order of selectable rows so the range
 * matches what the user sees on screen.
 */

import { SvelteSet } from 'svelte/reactivity';

interface UseMarqueeSelectionOptions {
	/** Latest selected-IDs set. Re-read per selection event so consumer-side reassignment works. */
	selectedIds: () => SvelteSet<string>;
	/** IDs in the current rendered order; used to compute shift+click ranges and gate marquee visibility. */
	orderedIds: () => string[];
	/** Document listeners attach only while the getter returns true. */
	enabled: () => boolean;
	/** DOM attribute key (after the `data-` prefix) that marks selectable rows. */
	attributeName?: () => string;
	/** Minimum pixel distance before a press becomes a marquee drag. */
	dragThresholdPx?: number;
}

export function useMarqueeSelection(options: UseMarqueeSelectionOptions) {
	const dragThresholdPx = options.dragThresholdPx ?? 5;

	let dragAnchorId = $state<string | null>(null);
	let isMarqueeDragging = $state(false);
	let mouseDownActive = false;
	let dragStartX = 0;
	let dragStartY = 0;
	let mousedownRowId: string | null = null;
	let dragMode: 'add' | 'remove' | null = null;
	let suppressNextClick = false;

	function resolveAttributeName(): string {
		return options.attributeName?.() ?? 'conversation-row';
	}

	/**
	 * `dataset` keys are camelCased. `data-conversation-row` -> `conversationRow`.
	 * We resolve the attribute name once per call and read via the camelCase key.
	 */
	function datasetKey(key: string = resolveAttributeName()): string {
		return key.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
	}

	function decideDragMode(startingRowId: string | null, currentlySelected: ReadonlySet<string>) {
		return startingRowId !== null && currentlySelected.has(startingRowId) ? 'remove' : 'add';
	}

	/**
	 * Range-select uses Finder-style toggle-by-target semantics: if the target
	 * row is currently selected the range becomes deselected, otherwise it
	 * becomes selected. Anchor moves to `toId` so chained shift+clicks keep
	 * extending from the previous endpoint.
	 */
	function rangeSelect(fromId: string, toId: string) {
		const selected = options.selectedIds();
		const order = options.orderedIds();
		const fromIdx = order.indexOf(fromId);
		const toIdx = order.indexOf(toId);
		if (fromIdx === -1 || toIdx === -1) return;
		const [lo, hi] = fromIdx < toIdx ? [fromIdx, toIdx] : [toIdx, fromIdx];
		const shouldSelect = !selected.has(toId);
		for (let i = lo; i <= hi; i++) {
			const id = order[i];
			if (shouldSelect) selected.add(id);
			else selected.delete(id);
		}
	}

	function findRowAtPoint(x: number, y: number): string | null {
		const attr = resolveAttributeName();
		const selector = `[data-${attr}]`;
		const key = datasetKey(attr);
		let bestMatch: HTMLElement | null = null;
		let bestCenterDistance = Infinity;

		for (const row of document.querySelectorAll<HTMLElement>(selector)) {
			const rect = row.getBoundingClientRect();
			if (y >= rect.top && y <= rect.bottom && x >= rect.left && x <= rect.right) {
				return row.dataset[key] ?? null;
			}
			if (x >= rect.left && x <= rect.right) {
				const centerDistance = Math.abs(y - (rect.top + rect.height / 2));
				if (centerDistance < bestCenterDistance) {
					bestCenterDistance = centerDistance;
					bestMatch = row;
				}
			}
		}
		return bestMatch ? (bestMatch.dataset[key] ?? null) : null;
	}

	function updateMarqueeRect(currentX: number, currentY: number) {
		const attr = resolveAttributeName();
		const selector = `[data-${attr}]`;
		const key = datasetKey(attr);
		const selected = options.selectedIds();
		const left = Math.min(dragStartX, currentX);
		const top = Math.min(dragStartY, currentY);
		const right = Math.max(dragStartX, currentX);
		const bottom = Math.max(dragStartY, currentY);
		const visibleIds = new SvelteSet(options.orderedIds());

		for (const row of document.querySelectorAll<HTMLElement>(selector)) {
			const id = row.dataset[key];
			if (!id || !visibleIds.has(id)) continue;

			const rect = row.getBoundingClientRect();
			const intersects = !(
				rect.right < left ||
				rect.left > right ||
				rect.bottom < top ||
				rect.top > bottom
			);

			if (dragMode === 'add') {
				if (intersects) selected.add(id);
			} else if (dragMode === 'remove') {
				if (intersects && selected.has(id)) selected.delete(id);
			}
		}
	}

	function handleDocumentMouseMove(event: MouseEvent) {
		if (!mouseDownActive) return;

		if (event.shiftKey && dragAnchorId !== null) {
			const target = findRowAtPoint(event.clientX, event.clientY);
			if (target && target !== mousedownRowId) rangeSelect(dragAnchorId, target);
			return;
		}

		if (!isMarqueeDragging) {
			const dx = event.clientX - dragStartX;
			const dy = event.clientY - dragStartY;
			if (Math.hypot(dx, dy) < dragThresholdPx) return;
			isMarqueeDragging = true;
			dragMode = decideDragMode(mousedownRowId, options.selectedIds());
		}
		updateMarqueeRect(event.clientX, event.clientY);
	}

	function handleDocumentMouseUp(event: MouseEvent) {
		if (isMarqueeDragging) {
			suppressNextClick = true;
			const target = findRowAtPoint(event.clientX, event.clientY);
			if (target) dragAnchorId = target;
		}
		isMarqueeDragging = false;
		mouseDownActive = false;
		mousedownRowId = null;
		dragMode = null;
		dragStartX = 0;
		dragStartY = 0;
	}

	function handleClickCapture(event: MouseEvent) {
		if (suppressNextClick) {
			event.stopPropagation();
			event.preventDefault();
			suppressNextClick = false;
		}
	}

	$effect(() => {
		if (!options.enabled()) {
			reset();
			return;
		}
		document.addEventListener('mousemove', handleDocumentMouseMove);
		document.addEventListener('mouseup', handleDocumentMouseUp);
		document.addEventListener('click', handleClickCapture, { capture: true });
		return () => {
			document.removeEventListener('mousemove', handleDocumentMouseMove);
			document.removeEventListener('mouseup', handleDocumentMouseUp);
			document.removeEventListener('click', handleClickCapture, { capture: true });
		};
	});

	function rowMouseDown(id: string, event: MouseEvent) {
		if (!options.enabled()) return;
		if (event.button !== 0) return;
		event.preventDefault();
		mouseDownActive = true;
		mousedownRowId = id;
		dragStartX = event.clientX;
		dragStartY = event.clientY;
		isMarqueeDragging = false;
		dragMode = null;
	}

	function rowClick(id: string, shiftKey: boolean) {
		if (!options.enabled()) return;
		const selected = options.selectedIds();

		if (shiftKey) {
			const anchor = dragAnchorId;
			if (anchor !== null && anchor !== id) {
				rangeSelect(anchor, id);
			} else if (selected.has(id)) {
				selected.delete(id);
			} else {
				selected.add(id);
			}
			dragAnchorId = id;
			return;
		}

		if (selected.has(id)) selected.delete(id);
		else selected.add(id);
		dragAnchorId = id;
	}

	function reset() {
		dragAnchorId = null;
		isMarqueeDragging = false;
		mouseDownActive = false;
		suppressNextClick = false;
		mousedownRowId = null;
		dragMode = null;
		dragStartX = 0;
		dragStartY = 0;
	}

	return {
		rowMouseDown,
		rowClick,
		reset,
		get dragAnchorId() {
			return dragAnchorId;
		}
	};
}
