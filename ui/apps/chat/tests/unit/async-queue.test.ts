import { createAsyncQueue } from '#lib/utils/async-queue.ts';
import { describe, expect, it } from 'vitest';

async function collect<T>(queue: AsyncIterable<T>): Promise<T[]> {
	const items: T[] = [];

	for await (const item of queue) items.push(item);

	return items;
}

describe('createAsyncQueue', () => {
	it('yields the items pushed before and during the iteration, then ends on close', async () => {
		const queue = createAsyncQueue<number>();

		queue.push(1);

		const collected = collect(queue);

		await Promise.resolve();
		queue.push(2);
		queue.push(3);
		queue.close();

		expect(await collected).toEqual([1, 2, 3]);
	});

	it('throws the failure after the pushed items are read', async () => {
		const queue = createAsyncQueue<number>();
		const seen: number[] = [];

		queue.push(1);
		queue.fail(new Error('cut'));

		await expect(
			(async () => {
				for await (const item of queue) seen.push(item);
			})()
		).rejects.toThrow('cut');
		expect(seen).toEqual([1]);
	});
});
