import {
	SIDEBAR_REVEAL_EDGE_PX,
	SIDEBAR_REVEAL_MAX_WIDTH,
	SIDEBAR_REVEAL_RATIO,
	SIDEBAR_REVEAL_SLOP_PX
} from '$lib/constants';
import { deviceStore, uiStore } from '$lib/stores';
import { remToPx } from '$lib/utils';

interface RevealGesture {
	startX: number;
	startY: number;
	pointerId: number;
	wasExpanded: boolean;
	locked: boolean;
}

/**
 * Phone sidebar reveal: a horizontal drag on the page slides it aside to uncover the sidebar.
 * A drag from the left edge opens it, and a drag or tap on the page closes it.
 */
export function useSidebarReveal() {
	/** Live page offset in px while a finger drags it, null when the store drives it. */
	let dragOffset = $state<number | null>(null);
	let gesture: RevealGesture | null = null;
	let suppressClick = false;

	function revealPx(): number {
		return Math.min(window.innerWidth * SIDEBAR_REVEAL_RATIO, remToPx(SIDEBAR_REVEAL_MAX_WIDTH));
	}

	function handlePointerDown(event: PointerEvent) {
		suppressClick = false;

		if (!deviceStore.isMobile || !event.isPrimary) return;

		const wasExpanded = uiStore.isSidebarExpanded;

		if (!wasExpanded && event.clientX > SIDEBAR_REVEAL_EDGE_PX) return;

		gesture = {
			locked: false,
			pointerId: event.pointerId,
			startX: event.clientX,
			startY: event.clientY,
			wasExpanded
		};
	}

	function handlePointerMove(event: PointerEvent) {
		if (!gesture || event.pointerId !== gesture.pointerId) return;

		const dx = event.clientX - gesture.startX;
		const dy = event.clientY - gesture.startY;

		if (!gesture.locked) {
			if (Math.abs(dx) < SIDEBAR_REVEAL_SLOP_PX && Math.abs(dy) < SIDEBAR_REVEAL_SLOP_PX) return;

			// a vertical move is a scroll, and a closed sidebar only opens to a rightward swipe
			if (Math.abs(dy) > Math.abs(dx) || (!gesture.wasExpanded && dx < 0)) {
				gesture = null;

				return;
			}

			gesture.locked = true;
			(event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
		}

		const max = revealPx();
		const start = gesture.wasExpanded ? max : 0;

		dragOffset = Math.min(max, Math.max(0, start + dx));
	}

	function handlePointerEnd(event: PointerEvent) {
		if (!gesture || event.pointerId !== gesture.pointerId) return;

		const { locked, wasExpanded } = gesture;

		gesture = null;

		if (!locked) {
			// a tap on the page while the sidebar is open closes it and is not a click on the page
			if (wasExpanded && event.type === 'pointerup') {
				suppressClick = true;
				uiStore.isSidebarExpanded = false;
			}

			return;
		}

		suppressClick = true;

		if (dragOffset !== null && event.type === 'pointerup') {
			uiStore.isSidebarExpanded = dragOffset > revealPx() / 2;
		}

		dragOffset = null;
	}

	/** Swallows the click that follows a drag or a tap that closed the sidebar. */
	function handleClickCapture(event: MouseEvent) {
		if (!suppressClick) return;

		suppressClick = false;
		event.stopPropagation();
		event.preventDefault();
	}

	return {
		get dragOffset() {
			return dragOffset;
		},
		handleClickCapture,
		handlePointerCancel: (event: PointerEvent) => handlePointerEnd(event),
		handlePointerDown,
		handlePointerMove,
		handlePointerUp: (event: PointerEvent) => handlePointerEnd(event)
	};
}
