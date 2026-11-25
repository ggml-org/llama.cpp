/**
 * The conversation's media in one slideshow: every image, audio clip and video
 * in the open thread, user attachments and tool results alike, in the order
 * they appear. A thumbnail anywhere in the chat opens it at its own item.
 */

import { conversationsStore } from './conversations/index.svelte';
import type { DatabaseMessageExtra } from '#lib/types/index.ts';
import { isMediaExtra } from '#lib/utils/index.ts';

/**
 * Two references to one item: the same object, or the same data copied over.
 * Reactive state hands out proxies, so the thumbnail's object is not always
 * identical to the one in the thread, and the data is the fallback.
 */
function isSameItem(a: DatabaseMessageExtra, b: DatabaseMessageExtra): boolean {
	if (a === b) return true;

	if (a.type !== b.type || a.name !== b.name) return false;

	const dataOf = (extra: DatabaseMessageExtra) =>
		'base64Url' in extra ? extra.base64Url : 'base64Data' in extra ? extra.base64Data : undefined;
	const dataA = dataOf(a);
	const dataB = dataOf(b);

	// lengths first: two different files of the same name almost never share
	// one, so the payloads themselves are compared only for a real match
	return dataA !== undefined && dataA.length === dataB?.length && dataA === dataB;
}

class MediaGalleryStore {
	index = $state(0);
	items = $state<DatabaseMessageExtra[]>([]);
	open = $state(false);

	/**
	 * Opens the slideshow of the open conversation at `item`. An item not in the
	 * thread ( a draft, another chat ) opens on its own.
	 */
	show(item: DatabaseMessageExtra): void {
		const items = conversationsStore.activeMessages.flatMap((message) =>
			(message.extra ?? []).filter(isMediaExtra)
		);
		const index = items.findIndex((candidate) => isSameItem(candidate, item));

		this.items = index >= 0 ? items : [item];
		this.index = Math.max(index, 0);
		this.open = true;
	}
}

export const mediaGalleryStore = new MediaGalleryStore();
