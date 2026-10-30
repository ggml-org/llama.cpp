import {
	SIDEBAR_REVEAL_COMMIT_RATIO,
	SIDEBAR_REVEAL_EDGE_PX,
	SIDEBAR_REVEAL_FLING_PX_PER_MS,
	SIDEBAR_REVEAL_MAX_WIDTH,
	SIDEBAR_REVEAL_MS,
	SIDEBAR_REVEAL_RATIO,
	SIDEBAR_REVEAL_SLOP_PX,
	SIDEBAR_REVEAL_SWIPE_PX
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
	/** Started away from the left edge, so it takes a longer swipe to open the sidebar. */
	isAnywhere: boolean;
	lastX: number;
	lastTime: number;
	/** Horizontal speed of the latest move in px/ms, positive to the right. */
	velocity: number;
}

/** Content a sideways drag already belongs to: text fields and anything that scrolls sideways. */
function ownsHorizontalDrag(target: EventTarget | null, root: Element): boolean {
	if (!(target instanceof Element)) return false;

	if (target.closest('input, textarea, [contenteditable="true"]')) return true;

	for (let el: Element | null = target; el && el !== root; el = el.parentElement) {
		const { overflowX } = getComputedStyle(el);

		if ((overflowX === 'auto' || overflowX === 'scroll') && el.scrollWidth > el.clientWidth) {
			return true;
		}
	}

	return false;
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
 * A drag from the left edge opens it, as does a longer rightward swipe from anywhere else, which
 * stays clear of the browser's own edge swipe for back and forward. A drag or tap on the page
 * closes it.
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
		deviceStore.isMobile &&
			(uiStore.isSidebarExpanded || dragOffset !== null || progress.current > 0)
	);

	// The slid page's transform makes it the box its fixed bars are placed in, rather than the
	// screen, which leaves them at the far ends of a scrolled page. The scroll is locked while it
	// is slid, so they are shifted back by how far the page was scrolled at its two ends.
	$effect(() => {
		const root = document.documentElement;

		if (isRevealed) {
			const scrollBottom = root.scrollHeight - window.scrollY - window.innerHeight;

			root.style.setProperty('--sidebar-reveal-pin-top', `${window.scrollY}px`);
			root.style.setProperty('--sidebar-reveal-pin-bottom', `${Math.max(0, scrollBottom)}px`);
		} else {
			root.style.removeProperty('--sidebar-reveal-pin-top');
			root.style.removeProperty('--sidebar-reveal-pin-bottom');
		}

		root.style.overflow = isRevealed ? 'hidden' : '';
	});

	// A pointer cannot stop a scroll, and a browser that has started one ignores any later
	// attempt, deciding on a touch's first move. So a move that leans sideways while a swipe to
	// open is pending, and every move of a swipe already under way, keeps the page from scrolling
	// under it; one that leans vertical is left alone, and the swipe gives way to the scroll.
	$effect(() => {
		if (!deviceStore.isMobile) return;

		function handleTouchMove(event: TouchEvent) {
			const touch = event.touches[0];

			if (!gesture || !touch || !event.cancelable) return;

			if (!gesture.locked) {
				if (gesture.wasExpanded) return;

				const dx = touch.clientX - gesture.startX;
				const dy = touch.clientY - gesture.startY;

				if (dx <= 0 || Math.abs(dx) <= Math.abs(dy)) return;
			}

			event.preventDefault();
		}

		window.addEventListener('touchmove', handleTouchMove, { passive: false });

		return () => window.removeEventListener('touchmove', handleTouchMove);
	});

	function handlePointerDown(event: PointerEvent) {
		suppressClick = false;

		if (!deviceStore.isMobile || !event.isPrimary) return;

		const wasExpanded = uiStore.isSidebarExpanded;
		const isAnywhere = !wasExpanded && event.clientX > SIDEBAR_REVEAL_EDGE_PX;

		if (isAnywhere && ownsHorizontalDrag(event.target, event.currentTarget as Element)) return;

		gesture = {
			isAnywhere,
			lastTime: event.timeStamp,
			lastX: event.clientX,
			locked: false,
			pointerId: event.pointerId,
			startX: event.clientX,
			startY: event.clientY,
			velocity: 0,
			wasExpanded
		};
	}

	function handlePointerMove(event: PointerEvent) {
		if (!gesture || event.pointerId !== gesture.pointerId) return;

		let dx = event.clientX - gesture.startX;

		const dy = event.clientY - gesture.startY;

		if (!gesture.locked) {
			const slop = gesture.isAnywhere ? SIDEBAR_REVEAL_SWIPE_PX : SIDEBAR_REVEAL_SLOP_PX;

			if (Math.abs(dx) < slop && Math.abs(dy) < slop) return;

			// a vertical move is a scroll, and a closed sidebar only opens to a rightward swipe; one
			// from anywhere must also lean sideways, as a slanted scroll is not a swipe
			const verticalShare = gesture.isAnywhere ? 1.5 : 1;

			if (Math.abs(dy) * verticalShare > Math.abs(dx) || (!gesture.wasExpanded && dx < 0)) {
				gesture = null;

				return;
			}

			gesture.locked = true;

			// the page follows the finger from here, rather than jumping by the swipe so far
			if (gesture.isAnywhere) {
				gesture.startX = event.clientX;
				dx = 0;
			}

			(event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
		}

		const elapsed = event.timeStamp - gesture.lastTime;

		if (elapsed > 0) gesture.velocity = (event.clientX - gesture.lastX) / elapsed;

		gesture.lastX = event.clientX;
		gesture.lastTime = event.timeStamp;

		const max = revealPx();
		const start = gesture.wasExpanded ? max : 0;

		dragOffset = Math.min(max, Math.max(0, start + dx));
	}

	function handlePointerEnd(event: PointerEvent) {
		if (!gesture || event.pointerId !== gesture.pointerId) return;

		const { locked, velocity, wasExpanded } = gesture;

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
			// a flick settles the way it was heading; otherwise a short pull is enough to move it
			// out of the state it started in
			if (Math.abs(velocity) >= SIDEBAR_REVEAL_FLING_PX_PER_MS) {
				uiStore.isSidebarExpanded = velocity > 0;
			} else {
				const pulled = wasExpanded ? revealPx() - dragOffset : dragOffset;

				uiStore.isSidebarExpanded =
					wasExpanded !== pulled > revealPx() * SIDEBAR_REVEAL_COMMIT_RATIO;
			}
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
