let _url = $state('#/');

export const settingsReferrer = {
	get url() {
		return _url;
	},
	set url(value: string) {
		_url = value;
	}
};
