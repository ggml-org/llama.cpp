/**
 * settingsReferrer - Remembers the settings route to return to after exit
 *
 * Tracks the last settings section the user was on so the app can return
 * there after a fallback exit. Standalone reactive value, no host.
 */

import { SETTINGS_FALLBACK_EXIT_ROUTE } from '$lib/constants';

let _url = $state<string>(SETTINGS_FALLBACK_EXIT_ROUTE);

export const settingsReferrer = {
	get url() {
		return _url;
	},
	set url(value: string) {
		_url = value;
	}
};
