// Shared state for the context gauge popup. The dial and the card live in
// different DOM subtrees, so open state and placement are coordinated here.
// centerX and bottom place the card just above the dial: both are measured
// once at open time, relative to the closest form ancestor; the dial and
// the card share that positioning frame, so the values stay exact for the
// whole time the card is open.
// Mouse pointers open on hover with a short grace delay to travel from
// dial to card; touch pointers toggle on tap.
let closeTimer: ReturnType<typeof setTimeout> | undefined;
let lastPointerType = '';

export const gaugePopup = $state({ open: false, centerX: 0, bottom: 0 });

const CARD_HALF_WIDTH = 128; // half of the w-64 card
const EDGE_MARGIN = 8;
const DIAL_GAP = 8;

function openFrom(trigger: HTMLElement): void {
	clearTimeout(closeTimer);
	const frame = trigger.closest('form');
	if (frame) {
		const frameRect = frame.getBoundingClientRect();
		const triggerRect = trigger.getBoundingClientRect();
		const centerX = triggerRect.left + triggerRect.width / 2 - frameRect.left;
		const min = CARD_HALF_WIDTH + EDGE_MARGIN;
		const max = frameRect.width - CARD_HALF_WIDTH - EDGE_MARGIN;
		gaugePopup.centerX = Math.min(Math.max(centerX, min), Math.max(min, max));
		gaugePopup.bottom = frameRect.bottom - triggerRect.top + DIAL_GAP;
	}
	gaugePopup.open = true;
}

function toggleFrom(trigger: HTMLElement): void {
	if (gaugePopup.open) {
		clearTimeout(closeTimer);
		gaugePopup.open = false;
	} else {
		openFrom(trigger);
	}
}

export function gaugePopupClose(): void {
	clearTimeout(closeTimer);
	gaugePopup.open = false;
}

export function gaugeTriggerPointerDown(event: PointerEvent): void {
	lastPointerType = event.pointerType;
}

export function gaugeTriggerClick(event: MouseEvent): void {
	if (lastPointerType !== 'touch') return;
	toggleFrom(event.currentTarget as HTMLElement);
}

export function gaugeTriggerKeydown(event: KeyboardEvent): void {
	if (event.key !== 'Enter' && event.key !== ' ') return;
	event.preventDefault();
	toggleFrom(event.currentTarget as HTMLElement);
}

export function gaugeTriggerEnter(event: PointerEvent): void {
	if (event.pointerType !== 'mouse') return;
	openFrom(event.currentTarget as HTMLElement);
}

export function gaugeTriggerLeave(event: PointerEvent): void {
	if (event.pointerType !== 'mouse') return;
	scheduleClose();
}

export function gaugeCardEnter(event: PointerEvent): void {
	if (event.pointerType !== 'mouse') return;
	clearTimeout(closeTimer);
}

export function gaugeCardLeave(event: PointerEvent): void {
	if (event.pointerType !== 'mouse') return;
	scheduleClose();
}

function scheduleClose(): void {
	clearTimeout(closeTimer);
	closeTimer = setTimeout(() => {
		gaugePopup.open = false;
	}, 150);
}
