/**
 * A queue read as an async iterable, to turn a callback API into a stream.
 * The iteration ends after `close`, and throws after `fail`, once the pushed items are read.
 */
export interface AsyncQueue<T> extends AsyncIterable<T> {
	push(item: T): void;
	close(): void;
	fail(error: unknown): void;
}

export function createAsyncQueue<T>(): AsyncQueue<T> {
	const items: T[] = [];

	let closed = false;
	let failure: { error: unknown } | undefined;
	let wake: (() => void) | undefined;

	const notify = () => {
		wake?.();
		wake = undefined;
	};

	return {
		close() {
			closed = true;
			notify();
		},
		fail(error) {
			failure = { error };
			notify();
		},
		push(item) {
			items.push(item);
			notify();
		},
		async *[Symbol.asyncIterator]() {
			while (true) {
				if (items.length > 0) {
					yield items.shift()!;
				} else if (failure) {
					throw failure.error;
				} else if (closed) {
					return;
				} else {
					await new Promise<void>((resolve) => (wake = resolve));
				}
			}
		}
	};
}
