/**
 * Keeps a scroll container pinned to its bottom while content grows at the end.
 *
 * The rules are deliberately small, because the browser already does the hard
 * part: content that grows below the viewport does not move the reader (scroll
 * anchoring), so a block they expand or a late layout pass leaves the viewport
 * exactly where it was. This only adds the one behaviour CSS cannot express -
 * following new content while the reader sits at the bottom - and it stops the
 * moment they scroll away from it.
 *
 * Given a `pin` at the end of the content, browsers with scroll anchoring
 * follow on their own: while sticking, the pin is the only anchor (see
 * `[data-stick-to-bottom]` in app.css), so every layout keeps it in place.
 * WebKit has no `overflow-anchor`, so there a resize observer writes the
 * scroll position instead, before the frame is painted.
 */

import { untrack } from 'svelte';

/** Distance from the bottom that still counts as being at the bottom. */
const AT_BOTTOM_PX = 24;
/** Distance from the bottom past which a way back down is offered. */
const FAR_FROM_BOTTOM_PX = 240;
/**
 * Growth this soon after a press is the reader's own doing (expanding a block,
 * opening a preview), so it must not drag the viewport down with it.
 */
const INTERACTION_GRACE_MS = 400;
/**
 * How fast a glide closes the distance to the bottom: the time constant of an
 * exponential ease, so it starts quickly and settles softly, and a target that
 * moves mid-way ( late layout, the first tokens ) bends the motion instead of
 * restarting it.
 */
const GLIDE_TIME_CONSTANT_MS = 90;
/** A glide animates at least this long, to still be running when the new rows land. */
const GLIDE_MIN_MS = 450;
/** A glide that has not settled after animating this long ends with a jump. */
const GLIDE_MAX_MS = 2000;
/**
 * After a glide, corrections this soon are glided too: measured heights and
 * fonts settle a moment after the rows land, and a jump there reads as a glitch.
 */
const GLIDE_SETTLE_MS = 2500;
/** A late frame advances the glide by no more than this, about two frames. */
const GLIDE_MAX_FRAME_MS = 34;
/** A glide ends once it has rested at the bottom this long. */
const GLIDE_SETTLED_MS = 150;
/** Close enough to the bottom for a glide to stop. */
const GLIDE_DONE_PX = 0.5;
const GLIDE_CANCEL_EVENTS = ['wheel', 'touchstart', 'keydown'] as const;
const GLIDE_SCROLL_KEYS: readonly string[] = [
	' ',
	'ArrowDown',
	'ArrowUp',
	'End',
	'Home',
	'PageDown',
	'PageUp'
];
/** Set on the scroller while the pin is its only scroll anchor. */
const STICK_ATTRIBUTE = 'data-stick-to-bottom';
const NATIVE_ANCHORING = typeof CSS !== 'undefined' && CSS.supports('overflow-anchor', 'none');

export class StickToBottom {
	/** Measured position: at the bottom, or within AT_BOTTOM_PX of it. */
	atBottom = $state(true);
	/** Far enough up that the "scroll to bottom" affordance is worth showing. */
	farFromBottom = $state(false);
	/** The reader is at the bottom, so new content should follow. */
	stick = $state(true);

	private cancelGlide = (event?: Event): void => {
		if (this.glideFrame === null) return;

		// typing the next message is not scrolling; only scroll keys outside a field are
		if (event instanceof KeyboardEvent) {
			const target = event.target as HTMLElement | null;
			const typing = target?.isContentEditable || target?.matches?.('input, textarea, select');

			if (typing || !GLIDE_SCROLL_KEYS.includes(event.key)) return;
		}

		this.stopGlide();
		this.glideSettleUntil = 0;
		this.measure();
	};
	private container: HTMLElement | undefined;
	private enabled = true;
	private glideFrame: number | null = null;
	private glideSettleUntil = 0;
	private graceTimer: ReturnType<typeof setTimeout> | undefined;
	private lastScrollHeight = 0;
	private lastScrollTop = 0;
	private markPressed = (): void => {
		this.pressedAt = performance.now();
		this.syncAnchor();
		clearTimeout(this.graceTimer);
		this.graceTimer = setTimeout(() => this.syncAnchor(), INTERACTION_GRACE_MS);
	};
	private native = false;
	private observer: ResizeObserver | null = null;
	// -Infinity, not 0: performance.now() starts at page load, so 0 would read
	// as a press during the first moments after a refresh, when a chat lands
	private pressedAt = -Infinity;

	private settleTimer: ReturnType<typeof setTimeout> | undefined;

	/**
	 * Binds the scroller and the element whose growth should be followed.
	 * `content` defaults to the scroller's first child, which is what a panel
	 * with a single growing body needs. `pin` is an empty element at the very
	 * end of the content that lets the browser do the following.
	 */
	attach(
		container: HTMLElement | undefined,
		content?: HTMLElement | undefined,
		pin?: HTMLElement | undefined
	): void {
		this.detach();
		this.container = container;
		this.native = NATIVE_ANCHORING && Boolean(pin);

		const observed = content ?? container?.firstElementChild;

		if (!container || !observed) return;

		this.observer = new ResizeObserver(() => this.follow());
		this.observer.observe(observed);
		container.addEventListener('pointerdown', this.markPressed, {
			capture: true,
			passive: true
		});

		this.lastScrollHeight = container.scrollHeight;
		this.lastScrollTop = container.scrollTop;
		// untracked: callers attach from an effect, and reading the state here
		// would re-attach on every flip and re-measure away a release
		untrack(() => this.measure());
	}

	detach(): void {
		this.stopGlide();
		this.glideSettleUntil = 0;
		clearTimeout(this.settleTimer);
		this.observer?.disconnect();
		this.observer = null;
		clearTimeout(this.graceTimer);
		this.container?.removeEventListener('pointerdown', this.markPressed, { capture: true });
		this.container?.removeAttribute(STICK_ATTRIBUTE);
		this.container = undefined;
	}

	/**
	 * Also follows `element`'s size changes, even right after a press: the grace keeps
	 * a block the reader expands from dragging the viewport, but the chat form
	 * expanding on their click must carry the conversation along. Returns the cleanup.
	 */
	followSizeOf(element: HTMLElement): () => void {
		const observer = new ResizeObserver(() => this.follow(true));

		observer.observe(element);

		return () => observer.disconnect();
	}

	/**
	 * Carries the reader to the bottom with a short eased scroll, for the moment
	 * a message is sent: the rows that land after this call slide into place
	 * instead of appearing there. The target is read again every frame, so
	 * content that is still settling bends the motion and never ends it with a
	 * jump. Any scroll input from the reader takes over at once.
	 *
	 * Why not `scrollToBottom('smooth')`: this is called before the new rows
	 * exist, so a native smooth scroll aims at the old bottom and ends there.
	 * When the rows land, `follow()` and the anchoring pin then snap to the new
	 * bottom in one frame, which is the jump this replaces; native smooth
	 * scrolling cannot be retargeted mid-flight, and it cannot hold either of
	 * those off. The press reset mirrors `scrollToBottom`: asking for the
	 * bottom outranks the press that led here.
	 */
	glideToBottom(): void {
		const el = this.container;

		if (!el) return;

		this.stick = true;
		this.pressedAt = -Infinity;

		if (
			!this.enabled ||
			typeof requestAnimationFrame === 'undefined' ||
			window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
		) {
			this.scrollToBottom();

			return;
		}

		this.atBottom = true;
		this.farFromBottom = false;
		this.glideSettleUntil = performance.now() + GLIDE_SETTLE_MS;
		// once the window is over the pin anchors again, without waiting for a scroll
		clearTimeout(this.settleTimer);
		this.settleTimer = setTimeout(() => this.syncAnchor(), GLIDE_SETTLE_MS + 1);
		this.startGlide();
	}

	/** Reader scrolls are the only thing that re-arms or stops following. */
	handleScroll(): void {
		this.measure();
	}

	scrollToBottom(behavior: ScrollBehavior = 'auto'): void {
		const el = this.container;

		if (!el) return;

		el.scrollTo({ behavior, top: el.scrollHeight });
		this.stick = true;
		this.atBottom = true;
		this.farFromBottom = false;
		// asking for the bottom outranks the press that led here (a sidebar row,
		// the send button), so the growth that follows is followed
		this.pressedAt = -Infinity;
		this.syncAnchor();
	}

	setEnabled(enabled: boolean): void {
		this.enabled = enabled;
		untrack(() => this.syncAnchor());
	}

	/**
	 * The document scroller's clientHeight is iOS's smaller viewport: Safari's with its
	 * toolbar shown, the installed app's without the status bar. The page scrolls on past it
	 * by that difference, so at its real bottom it would read as that far above it; the
	 * window's own height is the room it really shows.
	 */
	private distanceFromBottom(el: HTMLElement): number {
		return Math.max(0, el.scrollHeight - this.shownHeight(el) - el.scrollTop);
	}

	private follow(ignoreGrace = false): void {
		const el = this.container;

		if (!el) return;

		// a running glide reads the new bottom on its next frame
		if (this.glideFrame !== null) return;

		// shortly after a send, late layout is glided to as well
		if (
			this.enabled &&
			this.stick &&
			performance.now() < this.glideSettleUntil &&
			this.distanceFromBottom(el) > GLIDE_DONE_PX
		) {
			this.startGlide();

			return;
		}

		const pressedRecently =
			!ignoreGrace && performance.now() - this.pressedAt < INTERACTION_GRACE_MS;
		const atBottom = this.distanceFromBottom(el) <= AT_BOTTOM_PX;

		// with native anchoring the pin already held the bottom; write only when
		// it could not, e.g. once it went offscreen during an interaction grace
		if (this.enabled && this.stick && !pressedRecently && !(this.native && atBottom)) {
			el.scrollTop = el.scrollHeight;
			this.atBottom = true;
			this.farFromBottom = false;

			return;
		}

		this.atBottom = atBottom;
		this.farFromBottom = this.distanceFromBottom(el) > FAR_FROM_BOTTOM_PX;
	}

	/**
	 * Only the reader moving up stops following. Following only ever moves the
	 * viewport down, and content shrinking lowers scrollHeight, so moving up
	 * without shrinkage is the reader. It releases right away, even inside
	 * AT_BOTTOM_PX: a trackpad or smooth wheel scrolls in smaller steps, and
	 * staying stuck would snap each of them back while a response streams.
	 * The landing jump, anchoring adjustments and clamping must not unstick a
	 * reader who never touched the page, and none of them moves up that way.
	 */
	private measure(): void {
		const el = this.container;

		if (!el) return;

		const top = el.scrollTop;
		const height = el.scrollHeight;
		const distance = this.distanceFromBottom(el);

		// a glide is on its way to the bottom: its own steps are not the reader's
		if (this.glideFrame !== null) {
			this.lastScrollHeight = height;
			this.lastScrollTop = top;

			return;
		}

		this.atBottom = distance <= AT_BOTTOM_PX;
		this.farFromBottom = distance > FAR_FROM_BOTTOM_PX;

		if (top < this.lastScrollTop && height >= this.lastScrollHeight) {
			this.stick = false;
		} else if (this.atBottom) {
			this.stick = true;
		}

		this.lastScrollHeight = height;
		this.lastScrollTop = top;
		this.syncAnchor();
	}

	private shownHeight(el: HTMLElement): number {
		return el === document.documentElement
			? Math.max(el.clientHeight, window.innerHeight)
			: el.clientHeight;
	}

	private startGlide(): void {
		const el = this.container;

		if (!el || this.glideFrame !== null) return;

		let previous = performance.now();
		// time spent animating: a frame that arrives late ( the page was busy
		// rendering the new rows ) counts as one frame, not as the wait before it
		let elapsed = 0;
		let settledFor = 0;
		// kept apart from scrollTop, which rounds: steps under a pixel still add up
		let position = el.scrollTop;

		const step = (now: number): void => {
			const dt = Math.min(now - previous, GLIDE_MAX_FRAME_MS);
			const target = el.scrollHeight - this.shownHeight(el);

			previous = now;
			elapsed += dt;

			// the browser moved the page itself ( its own anchoring, a clamp ): go on from there
			if (Math.abs(el.scrollTop - position) > 1) position = el.scrollTop;

			const remaining = target - position;

			settledFor = Math.abs(remaining) <= GLIDE_DONE_PX ? settledFor + dt : 0;

			if (elapsed >= GLIDE_MAX_MS || (elapsed >= GLIDE_MIN_MS && settledFor >= GLIDE_SETTLED_MS)) {
				this.stopGlide();
				el.scrollTop = el.scrollHeight;
				this.measure();

				return;
			}

			position += remaining * (1 - Math.exp(-dt / GLIDE_TIME_CONSTANT_MS));
			el.scrollTop = position;
			this.glideFrame = requestAnimationFrame(step);
		};

		for (const type of GLIDE_CANCEL_EVENTS) {
			window.addEventListener(type, this.cancelGlide, { capture: true, passive: true });
		}

		this.glideFrame = requestAnimationFrame(step);
		// no pin anchoring while gliding: the browser would jump to it
		this.syncAnchor();
	}

	private stopGlide(): void {
		if (this.glideFrame === null) return;

		cancelAnimationFrame(this.glideFrame);
		this.glideFrame = null;

		for (const type of GLIDE_CANCEL_EVENTS) {
			window.removeEventListener(type, this.cancelGlide, { capture: true });
		}

		this.syncAnchor();
	}

	/**
	 * The pin anchors only while following, so a reader who scrolled up keeps their rows
	 * anchored. Not during a glide or the settle window after it either: there the glide
	 * does the following, and the pin would get there first with a jump.
	 */
	private syncAnchor(): void {
		const pressedRecently = performance.now() - this.pressedAt < INTERACTION_GRACE_MS;
		const gliding = this.glideFrame !== null || performance.now() < this.glideSettleUntil;

		this.container?.toggleAttribute(
			STICK_ATTRIBUTE,
			this.native && this.enabled && this.stick && !pressedRecently && !gliding
		);
	}
}

export function createStickToBottom(): StickToBottom {
	return new StickToBottom();
}
