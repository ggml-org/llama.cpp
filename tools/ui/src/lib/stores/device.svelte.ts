import { browser } from '$app/environment';
import { MEDIA_QUERIES } from '$lib/constants';

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

interface DeviceContext {
	/** Any iOS/iPadOS device, regardless of which app or browser embeds the page. */
	isIOSDevice: boolean;
	/** The Safari browser app on iOS, excluding other iOS browsers and WKWebViews. */
	isIOSSafari: boolean;
	/** Any WKWebView context on iOS: in-app browsers, embedded web views, and the
	 *  third-party iOS browsers (all of which share the WKWebView engine). */
	isWKWebView: boolean;
	/** PWA standalone mode: the page was launched from the home screen icon. */
	isStandalone: boolean;
}

const SERVER_DEFAULT: DeviceContext = {
	isIOSDevice: false,
	isIOSSafari: false,
	isWKWebView: false,
	isStandalone: false
};

function detect(): DeviceContext {
	if (!browser) return SERVER_DEFAULT;

	const ua = navigator.userAgent;
	const isTouch = navigator.maxTouchPoints > 0;

	const isIOSDevice = UA_PATTERNS.IOS_PHONE.test(ua) || (UA_PATTERNS.MACINTOSH.test(ua) && isTouch);

	// Safari keeps 'Safari/' in the UA; non-Safari iOS browsers emit their own
	// token instead. WKWebView typically omits 'Safari/' entirely.
	const hasSafariToken = UA_PATTERNS.SAFARI.test(ua) && !UA_PATTERNS.WEBVIEW_IOS.test(ua);
	const isIOSSafari = isIOSDevice && hasSafariToken;
	const isWKWebView = isIOSDevice && !hasSafariToken;

	// navigator.standalone is the legacy iOS-only flag (deprecated but still
	// present); display-mode: standalone is the modern standard (Safari 16.4+).
	const isStandalone =
		window.matchMedia(MEDIA_QUERIES.DISPLAY_MODE_STANDALONE).matches ||
		(navigator as Navigator & { standalone?: boolean }).standalone === true;

	return { isIOSDevice, isIOSSafari, isWKWebView, isStandalone };
}

export const device = $state<DeviceContext>(detect());

if (browser) {
	// isStandalone can change at runtime (e.g. user installs the PWA while the
	// tab is open); the UA-derived flags are static for the session.
	const mql = window.matchMedia(MEDIA_QUERIES.DISPLAY_MODE_STANDALONE);

	mql.addEventListener('change', (e) => {
		device.isStandalone = e.matches;
	});
}
