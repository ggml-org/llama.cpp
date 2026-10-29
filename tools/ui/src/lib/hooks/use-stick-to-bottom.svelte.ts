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
/**
 * Growth this soon after a press is the reader's own doing (expanding a block,
 * opening a preview), so it must not drag the viewport down with it.
 */
const INTERACTION_GRACE_MS = 400;
/** Set on the scroller while the pin is its only scroll anchor. */
const STICK_ATTRIBUTE = 'data-stick-to-bottom';
const NATIVE_ANCHORING = typeof CSS !== 'undefined' && CSS.supports('overflow-anchor', 'none');

export class StickToBottom {
	/** Measured position, which drives the "scroll to bottom" affordance. */
	atBottom = $state(true);
	/** The reader is at the bottom, so new content should follow. */
	stick = $state(true);

	private container: HTMLElement | undefined;
	private enabled = true;
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
		this.observer?.disconnect();
		this.observer = null;
		clearTimeout(this.graceTimer);
		this.container?.removeEventListener('pointerdown', this.markPressed, { capture: true });
		this.container?.removeAttribute(STICK_ATTRIBUTE);
		this.container = undefined;
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
		// asking for the bottom outranks the press that led here (a sidebar row,
		// the send button), so the growth that follows is followed
		this.pressedAt = -Infinity;
		this.syncAnchor();
	}

	setEnabled(enabled: boolean): void {
		this.enabled = enabled;
		untrack(() => this.syncAnchor());
	}

	private distanceFromBottom(el: HTMLElement): number {
		return el.scrollHeight - el.clientHeight - el.scrollTop;
	}

	private follow(): void {
		const el = this.container;

		if (!el) return;

		const pressedRecently = performance.now() - this.pressedAt < INTERACTION_GRACE_MS;
		const atBottom = this.distanceFromBottom(el) <= AT_BOTTOM_PX;

		// with native anchoring the pin already held the bottom; write only when
		// it could not, e.g. once it went offscreen during an interaction grace
		if (this.enabled && this.stick && !pressedRecently && !(this.native && atBottom)) {
			el.scrollTop = el.scrollHeight;
			this.atBottom = true;

			return;
		}

		this.atBottom = atBottom;
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

		this.atBottom = this.distanceFromBottom(el) <= AT_BOTTOM_PX;

		if (top < this.lastScrollTop && height >= this.lastScrollHeight) {
			this.stick = false;
		} else if (this.atBottom) {
			this.stick = true;
		}

		this.lastScrollHeight = height;
		this.lastScrollTop = top;
		this.syncAnchor();
	}

	/** The pin anchors only while following, so a reader who scrolled up keeps their rows anchored. */
	private syncAnchor(): void {
		const pressedRecently = performance.now() - this.pressedAt < INTERACTION_GRACE_MS;

		this.container?.toggleAttribute(
			STICK_ATTRIBUTE,
			this.native && this.enabled && this.stick && !pressedRecently
		);
	}
}

export function createStickToBottom(): StickToBottom {
	return new StickToBottom();
}
