/**
 * True for a mouse move the user made. A list that renders under a resting
 * pointer fires hover events too, and those must not move a picker's highlight.
 */
export function hasPointerMoved(event: MouseEvent) {
	return event.movementX !== 0 || event.movementY !== 0;
}
