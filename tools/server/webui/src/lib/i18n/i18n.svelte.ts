import { browser } from '$app/environment';
import { base } from '$app/paths';

const DEFAULT_LOCALE = 'en';
const LOCALE_STORAGE_KEY = 'LlamaCppWebui.locale';

type TranslationParams = Record<string, string | number>;
type Messages = Record<string, string>;
type Catalog = Record<string, Messages>;

function normalizeLocale(value: string): string {
	const trimmed = value.trim();
	if (!trimmed) return DEFAULT_LOCALE;

	const normalized = trimmed.replace('_', '-');
	const [language, region, ...rest] = normalized.split('-');

	if (!language) return DEFAULT_LOCALE;
	if (!region) return language.toLowerCase();

	const extra = rest.length > 0 ? `-${rest.join('-')}` : '';
	return `${language.toLowerCase()}-${region.toUpperCase()}${extra}`;
}

function getLocaleCandidates(locale: string): string[] {
	const normalized = normalizeLocale(locale);
	const baseLocale = normalized.split('-')[0] || DEFAULT_LOCALE;
	const candidates = [normalized];

	if (baseLocale && baseLocale !== normalized) {
		candidates.push(baseLocale);
	}

	if (!candidates.includes(DEFAULT_LOCALE)) {
		candidates.push(DEFAULT_LOCALE);
	}

	return candidates;
}

function interpolate(value: string, params?: TranslationParams): string {
	if (!params) return value;

	return value.replace(/\{(\w+)\}/g, (match, key) => {
		const replacement = params[key];
		return replacement === undefined ? match : String(replacement);
	});
}

class I18nStore {
	locale = $state(DEFAULT_LOCALE);
	messages = $state<Catalog>({});
	isReady = $state(false);
	isLoading = $state(false);
	private initialized = false;

	init() {
		if (!browser || this.initialized) return;
		this.initialized = true;

		const storedLocale = localStorage.getItem(LOCALE_STORAGE_KEY);
		const preferredLocale =
			storedLocale || navigator.languages?.[0] || navigator.language || DEFAULT_LOCALE;

		void this.setLocale(preferredLocale, { persist: true });
	}

	async setLocale(value: string, options?: { persist?: boolean }) {
		const candidates = getLocaleCandidates(value);
		const resolved = await this.loadFirstAvailable(candidates);
		this.locale = resolved;
		this.isReady = true;

		if (browser) {
			document.documentElement.lang = resolved;
			if (options?.persist !== false) {
				localStorage.setItem(LOCALE_STORAGE_KEY, resolved);
			}
		}
	}

	t(key: string, params?: TranslationParams): string {
		if (browser && !this.initialized) {
			this.init();
		}

		const primary = this.messages[this.locale];
		const fallback = this.messages[DEFAULT_LOCALE];
		const value = primary?.[key] ?? fallback?.[key] ?? key;
		return interpolate(value, params);
	}

	private async loadFirstAvailable(candidates: string[]): Promise<string> {
		for (const locale of candidates) {
			if (this.messages[locale]) {
				return locale;
			}

			if (await this.fetchLocale(locale)) {
				return locale;
			}
		}

		return DEFAULT_LOCALE;
	}

	private async fetchLocale(locale: string): Promise<boolean> {
		if (!browser) return false;

		this.isLoading = true;
		try {
			const response = await fetch(`${base}/locales/${locale}.json`, {
				cache: 'force-cache'
			});

			if (!response.ok) {
				return false;
			}

			const data = (await response.json()) as Messages;

			if (!data || typeof data !== 'object') {
				return false;
			}

			this.messages = { ...this.messages, [locale]: data };
			return true;
		} catch {
			return false;
		} finally {
			this.isLoading = false;
		}
	}
}

export const i18n = new I18nStore();
export const t = (key: string, params?: TranslationParams) => i18n.t(key, params);
