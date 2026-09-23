import { ChatActivityStore } from '$lib/stores/chat/activity.svelte';
import { beforeEach, describe, expect, it } from 'vitest';

describe('ChatActivityStore', () => {
	let store: ChatActivityStore;

	beforeEach(() => {
		store = new ChatActivityStore();
	});

	it('starts with no local or remote activity', () => {
		expect(store.loadingConvs).toEqual([]);
		expect(store.isLocal('a')).toBe(false);
		expect(store.isRemote('a')).toBe(false);
	});

	it('markLocal adds a conv to the local set and the loading union', () => {
		store.markLocal('a');

		expect(store.isLocal('a')).toBe(true);
		expect(store.isRemote('a')).toBe(false);
		expect(store.loadingConvs).toEqual(['a']);
	});

	it('localEnded removes a local conv', () => {
		store.markLocal('a');
		store.localEnded('a');

		expect(store.isLocal('a')).toBe(false);
		expect(store.loadingConvs).toEqual([]);
	});

	it('localEnded also drops a stale remote hint for the same conv', () => {
		store.markLocal('a');
		store.applyRemoteSnapshot(['a']);
		expect(store.isRemote('a')).toBe(true);

		store.localEnded('a');

		expect(store.isLocal('a')).toBe(false);
		expect(store.isRemote('a')).toBe(false);
		expect(store.loadingConvs).toEqual([]);
	});

	it('applyRemoteSnapshot adds remote convs and unions them with local', () => {
		store.markLocal('local');
		store.applyRemoteSnapshot(['remote']);

		expect(store.isRemote('remote')).toBe(true);
		expect(store.loadingConvs).toEqual(['local', 'remote']);
	});

	it('applyRemoteSnapshot removes remote convs missing from the snapshot', () => {
		store.applyRemoteSnapshot(['a', 'b']);
		store.applyRemoteSnapshot(['a']);

		expect(store.isRemote('a')).toBe(true);
		expect(store.isRemote('b')).toBe(false);
		expect(store.loadingConvs).toEqual(['a']);
	});

	it('applyRemoteSnapshot keeps local convs absent from the snapshot', () => {
		store.markLocal('local');
		store.applyRemoteSnapshot(['remote']);
		store.applyRemoteSnapshot([]);

		expect(store.isLocal('local')).toBe(true);
		expect(store.loadingConvs).toEqual(['local']);
	});

	it('loadingConvs does not duplicate a conv that is both local and remote', () => {
		store.markLocal('a');
		store.applyRemoteSnapshot(['a']);

		expect(store.loadingConvs).toEqual(['a']);
	});
});
