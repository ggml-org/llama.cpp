/**
 * Abort Signal Utilities
 *
 * Provides utilities for consistent AbortSignal propagation across the application.
 * These utilities help ensure that async operations can be properly cancelled
 * when needed (e.g., user stops generation, navigates away, etc.).
 */

/**
 * Throws an AbortError if the signal is aborted.
 * Use this at the start of async operations to fail fast.
 *
 * @param signal - Optional AbortSignal to check
 * @throws DOMException with name 'AbortError' if signal is aborted
 *
 * @example
 * ```ts
 * async function fetchData(signal?: AbortSignal) {
 *   throwIfAborted(signal);
 *   // ... proceed with operation
 * }
 * ```
 */
export function throwIfAborted(signal?: AbortSignal): void {
	if (signal?.aborted) {
		throw new DOMException('Operation was aborted', 'AbortError');
	}
}

/**
 * Checks if an error is an AbortError.
 * Use this to distinguish between user-initiated cancellation and actual errors.
 *
 * @param error - Error to check
 * @returns true if the error is an AbortError
 *
 * @example
 * ```ts
 * try {
 *   await fetchData(signal);
 * } catch (error) {
 *   if (isAbortError(error)) {
 *     // User cancelled - no error dialog needed
 *     return;
 *   }
 *   // Handle actual error
 * }
 * ```
 */
export function isAbortError(error: unknown): boolean {
	if (error instanceof DOMException && error.name === 'AbortError') {
		return true;
	}
	if (error instanceof Error && error.name === 'AbortError') {
		return true;
	}
	return false;
}

/**
 * Creates a new AbortController that is linked to one or more parent signals.
 * When any parent signal aborts, the returned controller also aborts.
 *
 * Useful for creating child operations that should be cancelled when
 * either the parent operation or their own timeout/condition triggers.
 *
 * @param signals - Parent signals to link to (undefined signals are ignored)
 * @returns A new AbortController linked to all provided signals
 *
 * @example
 * ```ts
 * // Link to user's abort signal and add a timeout
 * const linked = createLinkedController(userSignal, timeoutSignal);
 * await fetch(url, { signal: linked.signal });
 * ```
 */
export function createLinkedController(...signals: (AbortSignal | undefined)[]): AbortController {
	const controller = new AbortController();

	for (const signal of signals) {
		if (!signal) continue;

		// If already aborted, abort immediately
		if (signal.aborted) {
			controller.abort(signal.reason);
			return controller;
		}

		// Link to parent signal
		signal.addEventListener('abort', () => controller.abort(signal.reason), { once: true });
	}

	return controller;
}

/**
 * Creates an AbortSignal that times out after the specified duration.
 *
 * @param ms - Timeout duration in milliseconds
 * @returns AbortSignal that will abort after the timeout
 *
 * @example
 * ```ts
 * const signal = createTimeoutSignal(5000); // 5 second timeout
 * await fetch(url, { signal });
 * ```
 */
export function createTimeoutSignal(ms: number): AbortSignal {
	return AbortSignal.timeout(ms);
}

/**
 * Wraps a promise to reject if the signal is aborted.
 * Useful for making non-abortable promises respect an AbortSignal.
 *
 * @param promise - Promise to wrap
 * @param signal - AbortSignal to respect
 * @returns Promise that rejects with AbortError if signal aborts
 *
 * @example
 * ```ts
 * // Make a non-abortable operation respect abort signal
 * const result = await withAbortSignal(
 *   someNonAbortableOperation(),
 *   signal
 * );
 * ```
 */
export async function withAbortSignal<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
	if (!signal) return promise;

	throwIfAborted(signal);

	return new Promise<T>((resolve, reject) => {
		const abortHandler = () => {
			reject(new DOMException('Operation was aborted', 'AbortError'));
		};

		signal.addEventListener('abort', abortHandler, { once: true });

		promise
			.then((value) => {
				signal.removeEventListener('abort', abortHandler);
				resolve(value);
			})
			.catch((error) => {
				signal.removeEventListener('abort', abortHandler);
				reject(error);
			});
	});
}
