import {
	AUTO_SCROLL_AT_BOTTOM_THRESHOLD,
	AUTO_SCROLL_INTERVAL,
	USER_SCROLL_INTENT_MS
} from '$lib/constants';
import { KeyboardKey } from '$lib/enums';

const SCROLL_UP_KEYS: string[] = [KeyboardKey.ARROW_UP, KeyboardKey.HOME, KeyboardKey.PAGE_UP];

export interface AutoScrollOptions {
	disabled?: boolean;
}

/**
 * Creates an auto-scroll controller for a scrollable container.
 *
 * Features:
 * - Auto-scrolls to bottom during streaming/loading
 * - Stops auto-scroll when user manually scrolls up
 * - Resumes auto-scroll when user scrolls back to bottom
 *
 * Only the reader's own input (wheel, touch, scroll keys, scrollbar drag)
 * counts as scrolling up: the page also moves up on its own while a
 * response streams (scroll anchoring, blocks re-rendering shorter, the
 * send animation), and that must not stop following the response.
 */
export class AutoScrollController {
	private _autoScrollEnabled = $state(true);
	private _container: HTMLElement | undefined;
	private _disabled: boolean;
	private _draggingScrollbar = false;
	private _lastScrollTop = $state(0);
	private _lastUserIntentAt = 0;
	private _mutationObserver: MutationObserver | null = null;
	private _observerEnabled = false;
	private _rafPending = false;
	private _scrollInterval: ReturnType<typeof setInterval> | undefined;
	private _unbindUserIntent: (() => void) | null = null;
	private _userScrolledUp = $state(false);

	get autoScrollEnabled(): boolean {
		return this._autoScrollEnabled;
	}

	get userScrolledUp(): boolean {
		return this._userScrolledUp;
	}

	constructor(options: AutoScrollOptions = {}) {
		this._disabled = options.disabled ?? false;
	}

	/**
	 * Cleans up resources. Call this in onDestroy or when the component unmounts.
	 */
	destroy(): void {
		this.stopInterval();
		this._doStopObserving();
		this._unbindUserIntent?.();
		this._unbindUserIntent = null;
	}

	/**
	 * Enables auto-scroll (e.g., when user sends a message).
	 */
	enable(): void {
		if (this._disabled) return;

		this._userScrolledUp = false;
		this._autoScrollEnabled = true;
	}

	/**
	 * Handles scroll events to detect user scroll direction and toggle auto-scroll.
	 */
	handleScroll(): void {
		if (this._disabled || !this._container) return;

		const { clientHeight, scrollHeight, scrollTop } = this._container;
		const distanceFromBottom = scrollHeight - clientHeight - scrollTop;
		const isScrollingUp = scrollTop < this._lastScrollTop;
		const isAtBottom = distanceFromBottom < AUTO_SCROLL_AT_BOTTOM_THRESHOLD;
		const isUserScrolling =
			this._draggingScrollbar || performance.now() - this._lastUserIntentAt < USER_SCROLL_INTENT_MS;

		if (isScrollingUp && !isAtBottom && isUserScrolling) {
			this._userScrolledUp = true;
			this._autoScrollEnabled = false;
		} else if (isAtBottom && this._userScrolledUp) {
			this._userScrolledUp = false;
			this._autoScrollEnabled = true;
		}

		this._lastScrollTop = scrollTop;
	}

	/**
	 * Resets scroll state when switching conversations.
	 */
	resetScrollState(): void {
		this._userScrolledUp = false;
		this._autoScrollEnabled = !this._disabled;

		if (this._container) {
			this._lastScrollTop = this._container.scrollTop;
		}
	}

	/**
	 * Scrolls the container to the bottom instantly.
	 */
	scrollToBottom(): void {
		if (this._disabled || !this._container) return;

		this._container.scrollTop = this._container.scrollHeight;
	}

	/**
	 * Binds the controller to a scrollable container element.
	 */
	setContainer(container: HTMLElement | undefined): void {
		this._doStopObserving();
		this._unbindUserIntent?.();
		this._unbindUserIntent = null;
		this._container = container;

		if (container) {
			this._unbindUserIntent = this._bindUserIntent(container);
		}

		if (this._observerEnabled && container && !this._disabled) {
			this._doStartObserving();
		}
	}

	/**
	 * Updates the disabled state.
	 */
	setDisabled(disabled: boolean): void {
		if (this._disabled === disabled) return;

		this._disabled = disabled;

		if (disabled) {
			this._autoScrollEnabled = false;
			this.stopInterval();
			this._doStopObserving();
		} else if (this._observerEnabled && this._container && !this._mutationObserver) {
			this._doStartObserving();
		}
	}

	/**
	 * Starts the auto-scroll interval for continuous scrolling during streaming.
	 */
	startInterval(): void {
		if (this._disabled || this._scrollInterval) return;

		this._scrollInterval = setInterval(() => {
			this.scrollToBottom();
		}, AUTO_SCROLL_INTERVAL);
	}

	/**
	 * Starts a MutationObserver on the container that auto-scrolls to bottom
	 * on content changes. More responsive than interval-based polling.
	 */
	startObserving(): void {
		this._observerEnabled = true;

		if (this._container && !this._disabled && !this._mutationObserver) {
			this._doStartObserving();
		}
	}

	/**
	 * Stops the auto-scroll interval.
	 */
	stopInterval(): void {
		if (this._scrollInterval) {
			clearInterval(this._scrollInterval);
			this._scrollInterval = undefined;
		}
	}

	/**
	 * Stops the MutationObserver.
	 */
	stopObserving(): void {
		this._observerEnabled = false;
		this._doStopObserving();
	}

	/**
	 * Updates the auto-scroll interval based on streaming state.
	 * Call this in a $effect to automatically manage the interval.
	 */
	updateInterval(isStreaming: boolean): void {
		if (this._disabled) {
			this.stopInterval();

			return;
		}

		if (isStreaming && this._autoScrollEnabled) {
			if (!this._scrollInterval) {
				this.startInterval();
			}
		} else {
			this.stopInterval();
		}
	}

	/**
	 * Records the reader's own scroll input, so `handleScroll` can tell their
	 * scrolling from the page moving on its own.
	 */
	private _bindUserIntent(container: HTMLElement): () => void {
		const isPage = container === document.documentElement;
		const target: HTMLElement | Window = isPage ? window : container;
		const markIntent = () => {
			this._lastUserIntentAt = performance.now();
		};
		const onWheel = (event: WheelEvent) => {
			if (event.deltaY < 0) markIntent();
		};
		const onKeyDown = (event: KeyboardEvent) => {
			if (
				SCROLL_UP_KEYS.includes(event.key) ||
				(event.key === KeyboardKey.SPACE && event.shiftKey)
			) {
				markIntent();
			}
		};
		// a press right of the content box lands on the scrollbar
		const onPointerDown = (event: PointerEvent) => {
			const left = isPage ? 0 : container.getBoundingClientRect().left;

			this._draggingScrollbar = event.clientX >= left + container.clientWidth;
		};
		const onPointerUp = () => {
			this._draggingScrollbar = false;
		};

		target.addEventListener('wheel', onWheel as EventListener, { passive: true });
		target.addEventListener('touchmove', markIntent, { passive: true });
		target.addEventListener('keydown', onKeyDown as EventListener);
		target.addEventListener('pointerdown', onPointerDown as EventListener);
		window.addEventListener('pointerup', onPointerUp);

		return () => {
			target.removeEventListener('wheel', onWheel as EventListener);
			target.removeEventListener('touchmove', markIntent);
			target.removeEventListener('keydown', onKeyDown as EventListener);
			target.removeEventListener('pointerdown', onPointerDown as EventListener);
			window.removeEventListener('pointerup', onPointerUp);
		};
	}

	private _doStartObserving(): void {
		if (!this._container || this._mutationObserver) return;

		this._mutationObserver = new MutationObserver(() => {
			if (!this._autoScrollEnabled || this._rafPending) return;

			this._rafPending = true;
			requestAnimationFrame(() => {
				this._rafPending = false;

				if (this._autoScrollEnabled && this._container) {
					this._container.scrollTop = this._container.scrollHeight;
				}
			});
		});

		this._mutationObserver.observe(this._container, {
			characterData: true,
			childList: true,
			subtree: true
		});
	}

	private _doStopObserving(): void {
		if (this._mutationObserver) {
			this._mutationObserver.disconnect();
			this._mutationObserver = null;
		}

		this._rafPending = false;
	}
}

/**
 * Creates a new AutoScrollController instance.
 */
export function createAutoScrollController(options: AutoScrollOptions = {}): AutoScrollController {
	return new AutoScrollController(options);
}
