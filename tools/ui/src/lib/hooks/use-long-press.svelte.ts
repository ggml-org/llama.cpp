import { ROW_LONG_PRESS_MS, ROW_LONG_PRESS_SLOP_PX } from '$lib/constants';

interface LongPressOptions {
	/** Read on every press, so a hold only counts while this holds. */
	enabled: () => boolean;
	onLongPress: () => void;
}

interface Press {
	pointerId: number;
	startX: number;
	startY: number;
	timer: ReturnType<typeof setTimeout>;
}

/**
 * Touch press-and-hold: a finger that rests on an element fires onLongPress, and the click
 * that follows the release is swallowed. A drift past the slop is a scroll and cancels it.
 */
export function useLongPress({ enabled, onLongPress }: LongPressOptions) {
	/** True while a finger holds the element and the hold has not fired yet. */
	let isPressing = $state(false);
	let press: Press | null = null;
	let suppressClick = false;

	function cancel() {
		if (!press) return;

		clearTimeout(press.timer);
		press = null;
		isPressing = false;
	}

	function fire() {
		press = null;
		isPressing = false;
		suppressClick = true;
		navigator.vibrate?.(10);
		onLongPress();
	}

	function handlePointerDown(event: PointerEvent) {
		suppressClick = false;
		cancel();

		if (event.pointerType === 'mouse' || !event.isPrimary || !enabled()) return;

		press = {
			pointerId: event.pointerId,
			startX: event.clientX,
			startY: event.clientY,
			timer: setTimeout(fire, ROW_LONG_PRESS_MS)
		};
		isPressing = true;
	}

	function handlePointerMove(event: PointerEvent) {
		if (!press || event.pointerId !== press.pointerId) return;

		const drift = Math.hypot(event.clientX - press.startX, event.clientY - press.startY);

		if (drift > ROW_LONG_PRESS_SLOP_PX) cancel();
	}

	/** Swallows the click that follows a hold. */
	function handleClickCapture(event: MouseEvent) {
		if (!suppressClick) return;

		suppressClick = false;
		event.stopPropagation();
		event.preventDefault();
	}

	// Android raises its own context menu for a long touch, which stands in for the hold here
	function handleContextMenu(event: MouseEvent) {
		if (!press && !suppressClick) return;

		event.preventDefault();

		if (press) {
			clearTimeout(press.timer);
			fire();
		}
	}

	return {
		handleClickCapture,
		handleContextMenu,
		handlePointerDown,
		handlePointerEnd: cancel,
		handlePointerMove,
		get isPressing() {
			return isPressing;
		}
	};
}
