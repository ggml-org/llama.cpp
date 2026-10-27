/**
 * Keeps a scroll container pinned to its bottom while content grows at the end.
 *
 * The rules are deliberately small, because the browser already does the hard
 * part: content that grows below the viewport does not move the reader (scroll
 * anchoring), so a block they expand or a late layout pass leaves the viewport
 * exactly where it was. This only adds the one behaviour CSS cannot express -
 * following new content while the reader sits at the bottom - and it stops the
 * moment they scroll away from it.
 */

/** Distance from the bottom that still counts as being at the bottom. */
const AT_BOTTOM_PX = 24;
/**
 * Growth this soon after a press is the reader's own doing (expanding a block,
 * opening a preview), so it must not drag the viewport down with it.
 */
const INTERACTION_GRACE_MS = 400;

export class StickToBottom {
	/** Measured position, which drives the "scroll to bottom" affordance. */
	atBottom = $state(true);
	/** The reader is at the bottom, so new content should follow. */
	stick = $state(true);

	private container: HTMLElement | undefined;
	private enabled = true;
	private markPressed = (): void => {
		this.pressedAt = performance.now();
	};
	private observer: ResizeObserver | null = null;

	private pressedAt = 0;

	/**
	 * Binds the scroller and the element whose growth should be followed.
	 * `content` defaults to the scroller's first child, which is what a panel
	 * with a single growing body needs.
	 */
	attach(container: HTMLElement | undefined, content?: HTMLElement | undefined): void {
		this.detach();
		this.container = container;

		const observed = content ?? container?.firstElementChild;

		if (!container || !observed) return;

		this.observer = new ResizeObserver(() => this.follow());
		this.observer.observe(observed);
		container.addEventListener('pointerdown', this.markPressed, {
			capture: true,
			passive: true
		});

		this.measure();
	}

	detach(): void {
		this.observer?.disconnect();
		this.observer = null;
		this.container?.removeEventListener('pointerdown', this.markPressed, { capture: true });
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
	}

	setEnabled(enabled: boolean): void {
		this.enabled = enabled;
	}

	private distanceFromBottom(el: HTMLElement): number {
		return el.scrollHeight - el.clientHeight - el.scrollTop;
	}

	private follow(): void {
		const el = this.container;

		if (!el) return;

		const pressedRecently = performance.now() - this.pressedAt < INTERACTION_GRACE_MS;

		if (this.enabled && this.stick && !pressedRecently) {
			el.scrollTop = el.scrollHeight;
			this.atBottom = true;

			return;
		}

		this.atBottom = this.distanceFromBottom(el) <= AT_BOTTOM_PX;
	}

	private measure(): void {
		const el = this.container;

		if (!el) return;

		this.stick = this.atBottom = this.distanceFromBottom(el) <= AT_BOTTOM_PX;
	}
}

export function createStickToBottom(): StickToBottom {
	return new StickToBottom();
}
