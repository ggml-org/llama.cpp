import { AUTO_SCROLL_AT_BOTTOM_THRESHOLD, AUTO_SCROLL_INTERVAL } from '$lib/constants';

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
 */
export class AutoScrollController {
	private _autoScrollEnabled = $state(true);
	private _userScrolledUp = $state(false);
	private _lastScrollTop = $state(0);
	private _scrollInterval: ReturnType<typeof setInterval> | undefined;
	private _container: HTMLElement | undefined;
	private _disabled: boolean;
	private _mutationObserver: MutationObserver | null = null;
	private _rafPending = false;
	private _observerEnabled = false;
	constructor(options: AutoScrollOptions = {}) {
		this._disabled = options.disabled ?? false;
	}

	get autoScrollEnabled(): boolean {
		return this._autoScrollEnabled;
	}

	get userScrolledUp(): boolean {
		return this._userScrolledUp;
	}

	/**
	 * Binds the controller to a scrollable container element.
	 */
	setContainer(container: HTMLElement | undefined): void {
		this._doStopObserving();
		this._container = container;

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
	 * Handles scroll events to detect user scroll direction and toggle auto-scroll.
	 */
	handleScroll(): void {
		if (this._disabled || !this._container) return;

		const { scrollTop, scrollHeight, clientHeight } = this._container;
		const distanceFromBottom = scrollHeight - clientHeight - scrollTop;
		const isScrollingUp = scrollTop < this._lastScrollTop;
		const isAtBottom = distanceFromBottom < AUTO_SCROLL_AT_BOTTOM_THRESHOLD;

		if (isScrollingUp && !isAtBottom) {
			this._userScrolledUp = true;
			this._autoScrollEnabled = false;
		} else if (isAtBottom && this._userScrolledUp) {
			this._userScrolledUp = false;
			this._autoScrollEnabled = true;
		}

		this._lastScrollTop = scrollTop;
	}

	/**
	 * Scrolls the container to the bottom.
	 */
	scrollToBottom(behavior: ScrollBehavior = 'smooth'): void {
		if (this._disabled || !this._container) return;
		this._container.scrollTo({ top: this._container.scrollHeight, behavior });
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
	 * Starts the auto-scroll interval for continuous scrolling during streaming.
	 */
	startInterval(): void {
		if (this._disabled || this._scrollInterval) return;

		this._scrollInterval = setInterval(() => {
			this.scrollToBottom();
		}, AUTO_SCROLL_INTERVAL);
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
	 * Cleans up resources. Call this in onDestroy or when the component unmounts.
	 */
	destroy(): void {
		this.stopInterval();
		this._doStopObserving();
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
	 * Stops the MutationObserver.
	 */
	stopObserving(): void {
		this._observerEnabled = false;
		this._doStopObserving();
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
			childList: true,
			subtree: true,
			characterData: true
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
