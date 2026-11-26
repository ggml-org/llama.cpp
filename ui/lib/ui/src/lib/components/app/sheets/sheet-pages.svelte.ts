import { prefersReducedMotion } from 'svelte/motion';
import { fly, type TransitionConfig } from 'svelte/transition';
import { SHEET_PAGE_SLIDE_MS, SHEET_PAGE_SLIDE_PX } from '../../../constants';

/** A slide to the next page, or back to the previous one. */
enum SlideDirection {
	BACK = -1,
	FORWARD = 1
}

export interface SheetPagesState<P> {
	/** The page on top of the stack. */
	readonly page: P;
	/** Whether a page sits over the first one, so the header shows back instead of close. */
	readonly canGoBack: boolean;
	push(next: P): void;
	/** Back to the page below; does nothing on the first page. */
	pop(): void;
	/** Back to the first page, for each time the sheet opens. */
	reset(): void;
	/** The slide between pages: the next one comes in from the right, the previous one from the left. */
	transition(node: Element, params: { incoming: boolean }): TransitionConfig;
}

/**
 * A stack of pages inside a sheet, the native way: tapping a row pushes its page, back pops it.
 */
export function useSheetPages<P>(root: P): SheetPagesState<P> {
	let stack = $state<P[]>([root]);
	let direction = SlideDirection.FORWARD;

	return {
		get canGoBack() {
			return stack.length > 1;
		},
		get page() {
			return stack[stack.length - 1];
		},
		pop() {
			if (stack.length <= 1) return;

			direction = SlideDirection.BACK;
			stack = stack.slice(0, -1);
		},
		push(next: P) {
			direction = SlideDirection.FORWARD;
			stack = [...stack, next];
		},
		reset() {
			stack = [root];
		},
		transition(node: Element, { incoming }: { incoming: boolean }) {
			const offset = prefersReducedMotion.current ? 0 : SHEET_PAGE_SLIDE_PX;
			const sign = incoming ? direction : -direction;

			return fly(node, {
				duration: SHEET_PAGE_SLIDE_MS,
				opacity: 0,
				x: sign * offset
			});
		}
	};
}
