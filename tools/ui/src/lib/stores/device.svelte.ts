/**
 * deviceStore - Browser environment signals
 *
 * Device capabilities, OS theme and viewport in one class store:
 * deviceStore.isMobile, deviceStore.isIOSDevice / isIOSSafari / isWKWebView /
 * isStandalone, deviceStore.systemTheme.isDark.
 *
 * UA-derived flags are static for the session; isStandalone and systemTheme
 * track live media query changes.
 */

import { browser } from '$app/environment';
import { DEFAULT_MOBILE_BREAKPOINT, MEDIA_QUERIES } from '$lib/constants';
import { MediaQuery } from 'svelte/reactivity';

/**
 * iOS UA token detection.
 *
 * iPadOS 13+ ships a desktop macOS UA, so 'iPad' is no longer present in it;
 * a Macintosh UA combined with touch support is treated as an iPad instead.
 * Third-party iOS browsers (Chrome, Firefox, Edge) and in-app WKWebViews all
 * run on WKWebView and emit their own tokens (CriOS/FxiOS/EdgiOS/GSA) instead
 * of the trailing 'Safari/' the Safari app keeps.
 */
const UA_PATTERNS = {
	IOS_PHONE: /iPhone|iPod/,
	MACINTOSH: /Macintosh/,
	SAFARI: /Safari/,
	WEBVIEW_IOS: /CriOS|FxiOS|EdgiOS|GSA/
} as const;

class DeviceStore {
	/** Any iOS/iPadOS device, regardless of which app or browser embeds the page. */
	readonly isIOSDevice: boolean = false;
	/** The Safari browser app on iOS, excluding other iOS browsers and WKWebViews. */
	readonly isIOSSafari: boolean = false;
	/** PWA standalone mode: the page was launched from the home screen icon. */
	isStandalone = $state(false);
	/** Any WKWebView context on iOS: in-app browsers, embedded web views, and the
	 *  third-party iOS browsers (all of which share the WKWebView engine). */
	readonly isWKWebView: boolean = false;
	/** OS color scheme preference; the user override lives in settingsStore. */
	readonly systemTheme = $state({ isDark: false });

	private mobile = new MediaQuery(`max-width: ${DEFAULT_MOBILE_BREAKPOINT - 1}px`);

	get isMobile(): boolean {
		return this.mobile.current;
	}

	constructor() {
		if (!browser) return;

		const ua = navigator.userAgent;
		const isTouch = navigator.maxTouchPoints > 0;

		this.isIOSDevice =
			UA_PATTERNS.IOS_PHONE.test(ua) || (UA_PATTERNS.MACINTOSH.test(ua) && isTouch);
		// Safari keeps 'Safari/' in the UA; non-Safari iOS browsers emit their own
		// token instead. WKWebView typically omits 'Safari/' entirely.
		const hasSafariToken = UA_PATTERNS.SAFARI.test(ua) && !UA_PATTERNS.WEBVIEW_IOS.test(ua);

		this.isIOSSafari = this.isIOSDevice && hasSafariToken;
		this.isWKWebView = this.isIOSDevice && !hasSafariToken;
		// navigator.standalone is the legacy iOS-only flag (deprecated but still
		// present); display-mode: standalone is the modern standard (Safari 16.4+).
		this.isStandalone =
			window.matchMedia(MEDIA_QUERIES.DISPLAY_MODE_STANDALONE).matches ||
			(navigator as Navigator & { standalone?: boolean }).standalone === true;
		this.systemTheme.isDark = window.matchMedia(MEDIA_QUERIES.PREFERS_DARK).matches;

		// isStandalone and systemTheme can change at runtime (e.g. user installs the
		// PWA while the tab is open); the UA-derived flags are static for the session
		const standaloneMql = window.matchMedia(MEDIA_QUERIES.DISPLAY_MODE_STANDALONE);

		standaloneMql.addEventListener('change', (e) => {
			this.isStandalone = e.matches;
		});

		const darkMql = window.matchMedia(MEDIA_QUERIES.PREFERS_DARK);

		darkMql.addEventListener('change', (e) => {
			this.systemTheme.isDark = e.matches;
		});
	}
}

export const deviceStore = new DeviceStore();
