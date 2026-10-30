import {
	SIDEBAR_REVEAL_EDGE_PX,
	SIDEBAR_REVEAL_MAX_WIDTH,
	SIDEBAR_REVEAL_MS,
	SIDEBAR_REVEAL_RATIO,
	SIDEBAR_REVEAL_SLOP_PX
} from '$lib/constants';
import { deviceStore, uiStore } from '$lib/stores';
import { remToPx } from '$lib/utils';
import { untrack } from 'svelte';
import { quartOut } from 'svelte/easing';
import { Tween } from 'svelte/motion';

interface RevealGesture {
	startX: number;
	startY: number;
	pointerId: number;
	wasExpanded: boolean;
	locked: boolean;
}

const THEME_COLOR_SELECTOR = 'meta[name="theme-color"][data-sidebar-reveal]';

let colorContext: CanvasRenderingContext2D | null = null;

/** Resolves any CSS color (the computed oklab mix among them) to the plain rgb a theme-color takes. */
function toRgb(color: string): string {
	colorContext ??= document.createElement('canvas').getContext('2d', { willReadFrequently: true });

	if (!colorContext) return color;

	colorContext.clearRect(0, 0, 1, 1);
	colorContext.fillStyle = color;
	colorContext.fillRect(0, 0, 1, 1);

	const [r, g, b] = colorContext.getImageData(0, 0, 1, 1).data;

	return `rgb(${r}, ${g}, ${b})`;
}

/**
 * iOS Safari and the installed app tint their bars from theme-color, not from the page. While the
 * sidebar is open a theme-color of its own, ahead of the app's ones, follows the page background;
 * once it is closed again it goes, and the app's own theme-color applies as before.
 */
function syncThemeColor(progress: number) {
	let meta = document.head.querySelector<HTMLMetaElement>(THEME_COLOR_SELECTOR);

	if (progress === 0) {
		meta?.remove();

		return;
	}

	if (!meta) {
		meta = document.createElement('meta');
		meta.name = 'theme-color';
		meta.dataset.sidebarReveal = '';
		document.head.prepend(meta);
	}

	meta.content = toRgb(getComputedStyle(document.body).backgroundColor);
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

	// How far the sidebar is open, which the page and sidebar backgrounds follow (app.css) and
	// the browser bars with them. It is stepped every frame, since a theme-color cannot transition.
	const progress = new Tween(0, { duration: SIDEBAR_REVEAL_MS, easing: quartOut });

	$effect(() => {
		const dragProgress = dragOffset === null ? null : dragOffset / revealPx();
		const target = deviceStore.isMobile && uiStore.isSidebarExpanded ? 1 : 0;

		untrack(() => {
			if (dragProgress !== null) void progress.set(dragProgress, { duration: 0 });
			else void progress.set(target);
		});
	});

	$effect(() => {
		const value = progress.current;

		document.documentElement.style.setProperty('--sidebar-reveal-progress', String(value));
		syncThemeColor(value);
	});

	// the page stays locked from the moment it starts sliding aside until it is fully back
	const isRevealed = $derived(
		deviceStore.isMobile && (uiStore.isSidebarExpanded || progress.current > 0)
	);

	$effect(() => {
		document.documentElement.style.overflow = isRevealed ? 'hidden' : '';
	});

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
		handlePointerUp: (event: PointerEvent) => handlePointerEnd(event),
		/** The page is slid aside, even partly: its contents take no scrolling or taps. */
		get isRevealed() {
			return isRevealed;
		}
	};
}
