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
