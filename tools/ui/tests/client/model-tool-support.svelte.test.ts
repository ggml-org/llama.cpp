import { ServerRole } from '$lib/enums';
import { PropsService } from '$lib/services/props.service';
import { type ModelPropsHost, ModelPropsManager } from '$lib/stores/models/props.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import { flushSync } from 'svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const propsWithTemplate = (chat_template: string) => ({ chat_template }) as ApiLlamaCppServerProps;

describe('model tool support from server props', () => {
	let host: ModelPropsHost;
	let manager: ModelPropsManager;

	beforeEach(() => {
		host = {
			isModelLoaded: (id) => id === 'loaded',
			loadedModelIds: ['loaded'],
			models: [],
			selectedModelName: 'loaded'
		};
		manager = new ModelPropsManager(host);
		serverStore.role = ServerRole.MODEL;
		serverStore.props = null;
	});

	afterEach(() => {
		vi.restoreAllMocks();
		serverStore.clear();
	});

	it('uses the single-model template without a per-model request', () => {
		const fetch = vi.spyOn(PropsService, 'fetchForModel');

		serverStore.props = propsWithTemplate('{{ tools | tojson }}');
		expect(manager.checkModelSupportsToolUse('loaded')).toBe(true);
		serverStore.props = propsWithTemplate('{{ prompt }}');
		expect(manager.checkModelSupportsToolUse('loaded')).toBe(false);
		serverStore.props = null;
		expect(manager.checkModelSupportsToolUse('loaded')).toBe(false);
		expect(fetch).not.toHaveBeenCalled();
	});

	it('updates reactive consumers after router props arrive and reuses the cache', async () => {
		serverStore.role = ServerRole.ROUTER;
		const fetch = vi
			.spyOn(PropsService, 'fetchForModel')
			.mockResolvedValue(propsWithTemplate('{% for tool in tools %}'));

		let observed = false;

		const dispose = $effect.root(() => {
			$effect(() => {
				observed = manager.checkModelSupportsToolUse('loaded');
			});
		});

		try {
			flushSync();
			expect(observed).toBe(false);
			await expect.poll(() => observed).toBe(true);
			expect(manager.checkModelSupportsToolUse('loaded')).toBe(true);
			expect(fetch).toHaveBeenCalledExactlyOnceWith('loaded');
		} finally {
			dispose();
		}
	});

	it('does not fetch or borrow the global template for unloaded router models', () => {
		serverStore.role = ServerRole.ROUTER;
		serverStore.props = propsWithTemplate('{{ tools }}');
		const fetch = vi.spyOn(PropsService, 'fetchForModel');

		expect(manager.checkModelSupportsToolUse('unloaded')).toBe(false);
		expect(manager.checkModelSupportsToolUse('')).toBe(false);
		expect(fetch).not.toHaveBeenCalled();
	});

	it('does not retry failed props requests from a reactive selector', async () => {
		serverStore.role = ServerRole.ROUTER;
		const fetch = vi.spyOn(PropsService, 'fetchForModel').mockRejectedValue(new Error('offline'));
		const warning = vi.spyOn(console, 'warn').mockImplementation(() => {});

		let observed = true;

		const dispose = $effect.root(() => {
			const capabilities = $derived({
				reasoning: manager.checkModelSupportsThinking('loaded'),
				tools: manager.checkModelSupportsToolUse('loaded')
			});

			$effect(() => {
				observed = capabilities.tools;
			});
		});

		try {
			flushSync();
			await Promise.resolve();
			flushSync();
			await Promise.resolve();
			flushSync();
			expect(observed).toBe(false);
			expect(warning).toHaveBeenCalledOnce();
			expect(fetch).toHaveBeenCalledExactlyOnceWith('loaded');
		} finally {
			dispose();
		}
	});
});
