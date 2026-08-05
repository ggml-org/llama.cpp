import { KeyboardKey } from '$lib/enums';

/**
 * Shared keyboard navigation state for the chat-form pickers.
 *
 * Command, mention and working-directory pickers all track a highlighted
 * row (`hoveredIndex`) plus a `scrollTrigger` counter that the list uses to
 * scroll the active row into view on keyboard nav only. Arrow up/down move
 * the highlight, Escape closes, Enter selects. This hook owns that shared
 * state and key handling; each picker supplies its own list length, an
 * optional movement resolver (the command picker skips disabled commands),
 * and close/select callbacks.
 */
export interface UsePickerNavigationOptions {
	/** Whether the picker is open; gates all key handling. */
	isOpen: () => boolean;
	/** Current number of highlightable rows. */
	count: () => number;
	/**
	 * Resolve the row to highlight for a movement step, or -1 when no move
	 * is possible. Defaults to plain wraparound across `count()`.
	 */
	step?: (from: number, dir: 1 | -1) => number;
	/** Called on Escape. */
	onClose: () => void;
	/** Called on Enter when `hoveredIndex` points at a selectable row. */
	onSelect: (index: number) => void;
}

function wrapStep(from: number, dir: 1 | -1, count: number): number {
	return dir === 1 ? (from + 1) % count : from <= 0 ? count - 1 : from - 1;
}

export function usePickerNavigation(opts: UsePickerNavigationOptions) {
	let hoveredIndex = $state(-1);
	let scrollTrigger = $state(0);

	function resolve(from: number, dir: 1 | -1): number {
		const n = opts.count();
		if (n === 0) return -1;
		if (opts.step) return opts.step(from, dir);
		return wrapStep(from, dir, n);
	}

	function move(dir: 1 | -1) {
		const next = resolve(hoveredIndex, dir);
		if (next >= 0) {
			hoveredIndex = next;
			scrollTrigger++;
		}
	}

	/** Reset the highlight without bumping the scroll trigger (open / filter / result changes). */
	function reset(index: number) {
		hoveredIndex = index;
	}

	/** Bump the scroll trigger without moving the highlight (e.g. scroll a freshly prioritized list to top). */
	function bumpScroll() {
		scrollTrigger++;
	}

	/** Mouse hover highlights a row but must NOT bump the scroll trigger. */
	function setHover(index: number) {
		hoveredIndex = index;
	}

	/** Returns true when the key was consumed by the picker. */
	function handleKeydown(event: KeyboardEvent): boolean {
		if (!opts.isOpen()) return false;

		if (event.key === KeyboardKey.ESCAPE) {
			event.preventDefault();
			opts.onClose();
			return true;
		}

		if (event.key === KeyboardKey.ARROW_DOWN) {
			event.preventDefault();
			move(1);
			return true;
		}

		if (event.key === KeyboardKey.ARROW_UP) {
			event.preventDefault();
			move(-1);
			return true;
		}

		if (event.key === KeyboardKey.ENTER) {
			if (hoveredIndex >= 0 && hoveredIndex < opts.count()) {
				event.preventDefault();
				opts.onSelect(hoveredIndex);
				return true;
			}
			// No selectable row - let the caller's Enter-to-submit run.
			return false;
		}

		return false;
	}

	return {
		get hoveredIndex() {
			return hoveredIndex;
		},
		get scrollTrigger() {
			return scrollTrigger;
		},
		reset,
		setHover,
		move,
		bumpScroll,
		handleKeydown
	};
}

export type UsePickerNavigationReturn = ReturnType<typeof usePickerNavigation>;
