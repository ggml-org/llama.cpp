import { KeyboardKey } from '$lib/enums';

/**
 * Shared keyboard navigation state for the chat-form pickers: a highlighted
 * row, a scroll trigger, and Arrow/Escape/Enter handling.
 */
export interface UsePickerNavigationOptions {
	/** Gates all key handling. */
	isOpen: () => boolean;
	count: () => number;
	/**
	 * Resolve the row to highlight for a movement step, or -1 when no move
	 * is possible. Defaults to plain wraparound across `count()`.
	 */
	step?: (from: number, dir: 1 | -1) => number;
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

	/** Reset the highlight without bumping the scroll trigger. */
	function reset(index: number) {
		hoveredIndex = index;
	}

	/** Bump the scroll trigger without moving the highlight. */
	function bumpScroll() {
		scrollTrigger++;
	}

	/** Mouse hover highlights a row but must NOT bump the scroll trigger. */
	function setHover(index: number) {
		hoveredIndex = index;
	}

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
		bumpScroll,
		handleKeydown,
		get hoveredIndex() {
			return hoveredIndex;
		},
		move,
		reset,
		get scrollTrigger() {
			return scrollTrigger;
		},
		setHover
	};
}

export type UsePickerNavigationReturn = ReturnType<typeof usePickerNavigation>;
