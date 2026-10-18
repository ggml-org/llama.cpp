// Performance panel: the setting, the polling contract of perfStore, and the rendered figures on desktop and phone.

import ChatScreenHarness from './components/ChatScreenHarness.svelte';
import PerfDialogHarness from './components/PerfDialogHarness.svelte';
import PerfLayoutHarness from './components/PerfLayoutHarness.svelte';
import PerfDock from '$lib/components/app/perf/PerfDock.svelte';
import PerfPanel from '$lib/components/app/perf/PerfPanel.svelte';
import { CONFIG_LOCALSTORAGE_KEY, SETTING_CONFIG_DEFAULT, SETTINGS_KEYS } from '$lib/constants';
import { perfStore } from '$lib/stores/perf.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import type { ApiPerfResponse } from '$lib/types';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

const GIB = 1024 * 1024 * 1024;
const PHONE = { height: 844, width: 390 };
const DESKTOP = { height: 800, width: 1280 };
const WIDE = { height: 864, width: 1536 };
const PERF: ApiPerfResponse = {
	cpu: { busy_pct: 87, n_cpu: 28, proc_pct: 51 },
	gen: {
		generating: false,
		n_tokens: 64,
		tps_headline: 0.53,
		tps_last: 0.53,
		tps_total: 0.52,
		tps_window: null
	},
	gpu: { busy_pct: 3, mem_total_bytes: 12 * GIB, mem_used_bytes: 11.2 * GIB, name: 'RTX' },
	mem: {
		cached_bytes: 209 * GIB,
		free_bytes: 10 * GIB,
		occupied_bytes: 241 * GIB,
		total_bytes: 251 * GIB,
		used_bytes: 32 * GIB
	},
	nvme: { device: 'md0', read_bps: 4.2e9, read_bps_10s: 640e6 },
	predict: {
		actual: 100,
		guesses: 200,
		hits: 82,
		mode: 'prefetch',
		precision_total: 0.4,
		precision_window: 0.41,
		probes: 0,
		recall_total: 0.8,
		recall_window: 0.72,
		steps: 0
	},
	preload: {
		bytes_actual: 0,
		bytes_credited: GIB,
		bytes_hit: 2 * GIB,
		bytes_late: 0,
		bytes_pending: 1.5 * GIB,
		bytes_read_predictor: 4 * GIB,
		bytes_read_total: 10 * GIB,
		bytes_wasted: 2.4 * GIB,
		wasted_pct: 24
	},
	prompt: {
		layer: 42,
		n_layer: 93,
		n_tokens: 1600,
		processing: true,
		tps_headline: 14.8,
		tps_last: null,
		tps_total: null,
		tps_window: null
	},
	ts_ms: 0,
	window_s: 60
};

let perfCalls = 0;
let perfStatus = 200;
let hidden = false;

// /perf answers come from here (a plain response object: no stream reads outside the fake timers),
// everything else goes to the setup file's mock
function mockPerf(body: unknown = PERF) {
	perfCalls = 0;
	const spy = vi.spyOn(globalThis, 'fetch');
	const rest = spy.getMockImplementation();

	spy.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
		const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;

		if (!/\/perf\b/.test(url)) {
			return rest ? rest(input, init) : new Response('{}');
		}

		perfCalls++;
		const status = perfStatus;

		return {
			json: async () => (status === 200 ? body : { error: { message: 'unavailable' } }),
			ok: status === 200,
			status,
			statusText: ''
		} as Response;
	});
}

function setHidden(h: boolean) {
	hidden = h;
	document.dispatchEvent(new Event('visibilitychange'));
}

beforeEach(() => {
	hidden = false;
	perfStatus = 200;
	Object.defineProperty(document, 'hidden', { configurable: true, get: () => hidden });
});

afterEach(async () => {
	perfStore.setEnabled(false);
	vi.useRealTimers();
	await page.viewport(DESKTOP.width, DESKTOP.height);
});

describe('display performance setting', () => {
	it('defaults to off and persists when ticked', async () => {
		const { default: SettingsChatFields } =
			await import('$lib/components/app/settings/SettingsChat/SettingsChatFields.svelte');
		const { SETTINGS_REGISTRY } = await import('$lib/constants/settings.constants');

		localStorage.removeItem(CONFIG_LOCALSTORAGE_KEY);
		settingsStore.initialize();
		expect(SETTING_CONFIG_DEFAULT[SETTINGS_KEYS.DISPLAY_PERFORMANCE]).toBe(false);
		expect(settingsStore.config[SETTINGS_KEYS.DISPLAY_PERFORMANCE]).toBe(false);

		const fields = SETTINGS_REGISTRY.flatMap((s) => s.settings).filter(
			(f) => f.key === SETTINGS_KEYS.DISPLAY_PERFORMANCE
		);
		const screen = render(SettingsChatFields, {
			fields,
			localConfig: settingsStore.config,
			onConfigChange: (key: string, value: string | boolean) =>
				settingsStore.updateConfig(key as keyof typeof settingsStore.config, value as never)
		});
		const box = screen.getByRole('checkbox', { name: 'Display performance' });

		await expect.element(box).not.toBeChecked();
		await box.click();

		const stored = JSON.parse(localStorage.getItem(CONFIG_LOCALSTORAGE_KEY) ?? '{}');

		expect(stored.displayPerformance).toBe(true);
		settingsStore.updateConfig(SETTINGS_KEYS.DISPLAY_PERFORMANCE, false);
	});
});

describe('perfStore polling', () => {
	it('fetches at once when enabled, then once a second', async () => {
		vi.useFakeTimers();
		mockPerf();
		perfStore.setEnabled(true);
		await vi.advanceTimersByTimeAsync(0);
		expect(perfCalls).toBe(1);
		await vi.advanceTimersByTimeAsync(3000);
		expect(perfCalls).toBe(4);
		expect(perfStore.data?.gen.tps_headline).toBe(0.53);
	});

	it('does not fetch while hidden, and resumes at once when visible', async () => {
		vi.useFakeTimers();
		mockPerf();
		perfStore.setEnabled(true);
		await vi.advanceTimersByTimeAsync(0);
		setHidden(true);
		await vi.advanceTimersByTimeAsync(5000);
		expect(perfCalls).toBe(1);
		setHidden(false);
		await vi.advanceTimersByTimeAsync(0);
		expect(perfCalls).toBe(2);
	});

	it('stops when disabled', async () => {
		vi.useFakeTimers();
		mockPerf();
		perfStore.setEnabled(true);
		await vi.advanceTimersByTimeAsync(1000);
		const n = perfCalls;

		perfStore.setEnabled(false);
		await vi.advanceTimersByTimeAsync(10000);
		expect(perfCalls).toBe(n);
		expect(perfStore.data).toBeNull();
	});

	it('keeps polling on 503, every 5 s after 3 failures', async () => {
		vi.useFakeTimers();
		perfStatus = 503;
		mockPerf();
		perfStore.setEnabled(true);
		await vi.advanceTimersByTimeAsync(0);
		await vi.advanceTimersByTimeAsync(2000);
		expect(perfCalls).toBe(3);
		await vi.advanceTimersByTimeAsync(4900);
		expect(perfCalls).toBe(3);
		await vi.advanceTimersByTimeAsync(100);
		expect(perfCalls).toBe(4);
		await vi.advanceTimersByTimeAsync(5000);
		expect(perfCalls).toBe(5);
		expect(perfStore.unavailable).toBe(false);

		// the model is loaded: back to once a second
		perfStatus = 200;
		await vi.advanceTimersByTimeAsync(5000);
		expect(perfCalls).toBe(6);
		await vi.advanceTimersByTimeAsync(1000);
		expect(perfCalls).toBe(7);
	});

	it('fetches at once when re-enabled during a fetch of the previous session', async () => {
		vi.useFakeTimers();
		mockPerf();
		const spy = vi.mocked(globalThis.fetch);
		const answer = spy.getMockImplementation()!;

		let release: () => void = () => {};
		let held = 0;

		// the first /perf request hangs until released
		spy.mockImplementation((input, init) => {
			if (held++ === 0) {
				return new Promise((resolve) => {
					release = () => resolve(answer(input, init));
				});
			}

			return answer(input, init);
		});
		perfStore.setEnabled(true);
		await vi.advanceTimersByTimeAsync(0);
		expect(held).toBe(1);
		perfStore.setEnabled(false);
		perfStore.setEnabled(true);
		await vi.advanceTimersByTimeAsync(0);
		expect(held).toBe(1); // still in flight: no second request
		release();
		await vi.advanceTimersByTimeAsync(0);
		expect(held).toBe(2); // the new session's first fetch follows at once, not after 1 s
		await vi.advanceTimersByTimeAsync(1000);
		expect(held).toBe(3);
	});

	it('ignores a 404 that answers an earlier session', async () => {
		vi.useFakeTimers();
		mockPerf();
		const spy = vi.mocked(globalThis.fetch);
		const answer = spy.getMockImplementation()!;

		let release: () => void = () => {};
		let n = 0;

		// the first request is held, then answers 404 (as a server without /perf would)
		spy.mockImplementation((input, init) => {
			if (n++ === 0) {
				return new Promise((resolve) => {
					release = () =>
						resolve({
							json: async () => ({ error: { message: 'Not Found' } }),
							ok: false,
							status: 404,
							statusText: ''
						} as Response);
				});
			}

			return answer(input, init);
		});
		perfStore.setEnabled(true);
		await vi.advanceTimersByTimeAsync(0);
		perfStore.setEnabled(false);
		perfStatus = 200;
		perfStore.setEnabled(true);
		release(); // the first session's request answers 404
		await vi.advanceTimersByTimeAsync(0);
		expect(perfStore.unavailable).toBe(false);
		await vi.advanceTimersByTimeAsync(1000);
		expect(perfStore.data?.cpu?.busy_pct).toBe(87);
	});

	it('stops on 404 (a server without /perf)', async () => {
		vi.useFakeTimers();
		perfStatus = 404;
		mockPerf();
		perfStore.setEnabled(true);
		await vi.advanceTimersByTimeAsync(10000);
		expect(perfCalls).toBe(1);
		expect(perfStore.unavailable).toBe(true);
	});
});

describe('performance panel', () => {
	it('shows the 9 figures in order, the last request speed when the window is empty', async () => {
		perfStore.data = PERF;
		const screen = render(PerfPanel);
		const section = screen.getByRole('region', { name: 'Performance' });

		await expect.element(section).toBeVisible();

		const values = Array.from(document.querySelectorAll('dd[data-perf]')).map((e) => [
			e.getAttribute('data-perf'),
			e.textContent?.trim()
		]);

		expect(values).toEqual([
			['CPU', '87%'],
			['GPU', '3%'],
			['RAM', '241 / 251 GB'],
			['VRAM', '11.2 / 12 GB'],
			['gen', '0.53 t/s'],
			['pp', '14.8 t/s'],
			['NVMe', '4.2 GB/s'],
			['guesses right', '41%'],
			['preload wasted', '24%']
		]);

		// expand: the details and the definition of wasted
		await screen.getByRole('button', { name: /^Performance (details|less)$/ }).click();
		await expect.element(screen.getByText('layer 42/93', { exact: false })).toBeVisible();
		await expect.element(screen.getByText('10 s 640 MB/s, md0')).toBeVisible();
		await expect.element(screen.getByText('Preload wasted:', { exact: false })).toBeVisible();
		perfStore.data = null;
	});

	it('prints n/a for unknown figures', async () => {
		perfStore.data = { ...PERF, gpu: null, nvme: null, predict: null, preload: null };
		render(PerfPanel);
		const get = (k: string) => document.querySelector(`dd[data-perf="${k}"]`)?.textContent?.trim();

		expect(get('GPU')).toBe('n/a');
		expect(get('VRAM')).toBe('n/a');
		expect(get('NVMe')).toBe('n/a');
		expect(get('guesses right')).toBe('n/a');
		expect(get('preload wasted')).toBe('n/a');
		perfStore.data = null;
	});

	it('shows ... for the generation speed until the second token of the first request', async () => {
		perfStore.data = {
			...PERF,
			gen: { ...PERF.gen, generating: true, n_tokens: 1, tps_headline: null, tps_last: null }
		};
		render(PerfPanel);
		await vi.waitFor(() =>
			expect(document.querySelector('[data-perf="gen"]')?.textContent?.trim()).toBe('...')
		);
		perfStore.data = null;
	});

	it('marks the figures stale when the last poll failed', async () => {
		perfStore.data = PERF;
		perfStore.error = 'Loading model';
		const screen = render(PerfPanel);

		await expect.element(screen.getByText('stale')).toBeVisible();
		expect(document.querySelector('section[aria-label="Performance"]')?.className).toContain(
			'opacity-60'
		);
		perfStore.error = null;
		perfStore.data = null;
	});

	it('renders a strip above the composer on a phone, all 9 values in 390 px, expanded by a tap', async () => {
		await page.viewport(PHONE.width, PHONE.height);
		mockPerf();
		perfStore.setEnabled(true);

		const screen = render(ChatScreenHarness);
		const toggle = screen.getByRole('button', { name: /^Performance (details|less)$/ });

		await expect.element(toggle).toHaveAttribute('aria-expanded', 'false');
		await expect.element(screen.getByText('0.53 t/s')).toBeVisible();

		// every figure inside the screen, no horizontal overflow
		const values = Array.from(document.querySelectorAll('[data-perf]')) as HTMLElement[];

		expect(values.length).toBe(9);
		for (const v of values) {
			const r = v.getBoundingClientRect();

			expect(r.width).toBeGreaterThan(0);
			expect(r.left).toBeGreaterThanOrEqual(0);
			expect(r.right).toBeLessThanOrEqual(PHONE.width);
		}
		expect(document.documentElement.scrollWidth).toBeLessThanOrEqual(PHONE.width);

		// above the composer, not over it
		const form = document.querySelector('.conversation-chat-form') as HTMLElement;
		const section = document.querySelector('section[aria-label="Performance"]') as HTMLElement;

		expect(section.getBoundingClientRect().bottom).toBeLessThanOrEqual(
			form.getBoundingClientRect().top + 1
		);

		// the fixed composer covers the bottom of the messages: the padding covers the strip too
		const strip = section.parentElement as HTMLElement;
		const pad = parseFloat(
			getComputedStyle(document.querySelector('.chat-screen') as HTMLElement).paddingBottom
		);

		expect(pad).toBeCloseTo(192 + strip.getBoundingClientRect().height, 0);

		await toggle.click();
		await expect.element(toggle).toHaveAttribute('aria-expanded', 'true');
		await expect.element(screen.getByText('layer 42/93', { exact: false })).toBeVisible();
	});

	for (const vp of [
		{ height: 768, width: 1024 },
		{ height: 800, width: 1280 }
	]) {
		it(`uses the strip next to an open sidebar at ${vp.width}x${vp.height}, the send button stays clickable`, async () => {
			await page.viewport(vp.width, vp.height);
			mockPerf();
			settingsStore.updateConfig(SETTINGS_KEYS.DISPLAY_PERFORMANCE, true);
			render(PerfLayoutHarness);
			await vi.waitFor(() => expect(perfCalls).toBeGreaterThan(0));
			await vi.waitFor(() =>
				expect(document.querySelectorAll('section[aria-label="Performance"]').length).toBe(1)
			);

			const form = document.querySelector('.conversation-chat-form') as HTMLElement;
			const input = form.querySelector('[contenteditable="true"], textarea') as HTMLElement;

			// with some text the send button is enabled, so the hit test must land on the button itself
			await userEvent.click(input);
			await userEvent.keyboard('x');
			await vi.waitFor(() =>
				expect((form.querySelector('button[type="submit"]') as HTMLButtonElement).disabled).toBe(
					false
				)
			);
			const send = form.querySelector('button[type="submit"]') as HTMLElement;
			const r = send.getBoundingClientRect();
			const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);

			expect(send.contains(hit)).toBe(true);
			settingsStore.updateConfig(SETTINGS_KEYS.DISPLAY_PERFORMANCE, false);
		});
	}

	it('puts the card in the right margin of a wide screen: clear of the composer, clicks go through', async () => {
		await page.viewport(WIDE.width, WIDE.height);
		mockPerf();
		let underClicks = 0;

		settingsStore.updateConfig(SETTINGS_KEYS.DISPLAY_PERFORMANCE, true);
		render(PerfLayoutHarness, { onUnderClick: () => underClicks++, showUnder: true });
		await vi.waitFor(() =>
			expect(document.querySelector('[data-perf="CPU"]')?.textContent).toBe('87%')
		);

		const sections = document.querySelectorAll('section[aria-label="Performance"]');

		expect(sections.length).toBe(1); // the card, no strip
		const card = sections[0].getBoundingClientRect();
		const form = (
			document.querySelector('.conversation-chat-form') as HTMLElement
		).getBoundingClientRect();

		expect(card.left >= form.right || card.bottom <= form.top).toBe(true);
		expect(getComputedStyle(sections[0].parentElement as HTMLElement).zIndex).toBe('40');

		// a click on the card's figures lands on what is underneath
		const under = document.querySelector('[data-testid="under"]') as HTMLElement;
		const hit = document.elementFromPoint(card.left + card.width / 2, card.top + card.height / 3);

		expect(hit).toBe(under);
		(hit as HTMLElement).click();
		expect(underClicks).toBe(1);
		settingsStore.updateConfig(SETTINGS_KEYS.DISPLAY_PERFORMANCE, false);
	});
});

describe('performance card under dialogs', () => {
	it('an open settings dialog covers the card at 1600x900', async () => {
		await page.viewport(1600, 900);
		mockPerf();
		settingsStore.updateConfig(SETTINGS_KEYS.DISPLAY_PERFORMANCE, true);
		render(PerfDialogHarness, { open: true });
		await vi.waitFor(() =>
			expect(document.querySelector('[data-perf="CPU"]')?.textContent).toBe('87%')
		);
		await vi.waitFor(() =>
			expect(document.querySelector('[data-slot="dialog-overlay"]')).not.toBeNull()
		);

		// the card lets clicks through, so a plain hit test says nothing about paint order: let it take clicks for the probe
		const section = document.querySelector('section[aria-label="Performance"]') as HTMLElement;

		section.style.pointerEvents = 'auto';
		(section.parentElement as HTMLElement).style.pointerEvents = 'auto';
		const card = section.getBoundingClientRect();
		const hit = document.elementFromPoint(
			card.left + card.width / 2,
			card.top + card.height / 2
		) as HTMLElement;

		expect(hit.closest('section[aria-label="Performance"]')).toBeNull();
		expect(
			hit.closest('[data-slot="dialog-overlay"], [data-slot="dialog-content"]')
		).not.toBeNull();
		settingsStore.updateConfig(SETTINGS_KEYS.DISPLAY_PERFORMANCE, false);
	});
});

describe('setting wiring', () => {
	afterEach(() => {
		// mockUiSettings shadows the store's getter on the instance
		Reflect.deleteProperty(serverStore, 'props');
	});

	it('the dock switches perfStore with the setting', async () => {
		mockPerf();
		settingsStore.updateConfig(SETTINGS_KEYS.DISPLAY_PERFORMANCE, false);
		render(PerfDock);
		expect(perfStore.enabled).toBe(false);
		settingsStore.updateConfig(SETTINGS_KEYS.DISPLAY_PERFORMANCE, true);
		await vi.waitFor(() => expect(perfStore.enabled).toBe(true));
		settingsStore.updateConfig(SETTINGS_KEYS.DISPLAY_PERFORMANCE, false);
		await vi.waitFor(() => expect(perfStore.enabled).toBe(false));
	});

	function mockUiSettings(ui: Record<string, boolean>) {
		Object.defineProperty(serverStore, 'props', {
			configurable: true,
			get: () =>
				({
					default_generation_settings: { params: { temperature: 0.8 } },
					ui_settings: ui
				}) as unknown as typeof serverStore.props
		});
	}

	it('--ui-config displayPerformance applies on a first visit only', () => {
		localStorage.removeItem(CONFIG_LOCALSTORAGE_KEY);
		settingsStore.initialize();
		mockUiSettings({ displayPerformance: true });
		settingsStore.syncWithServerDefaults();
		expect(settingsStore.config[SETTINGS_KEYS.DISPLAY_PERFORMANCE]).toBe(true);

		// a browser with a saved config keeps its choice
		localStorage.setItem(CONFIG_LOCALSTORAGE_KEY, JSON.stringify({ displayPerformance: false }));
		settingsStore.initialize();
		settingsStore.syncWithServerDefaults();
		expect(settingsStore.config[SETTINGS_KEYS.DISPLAY_PERFORMANCE]).toBe(false);
		localStorage.removeItem(CONFIG_LOCALSTORAGE_KEY);
		settingsStore.initialize();
	});
});
