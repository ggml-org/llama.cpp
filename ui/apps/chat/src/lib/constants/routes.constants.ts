/** Query params the chat routes read from the URL. */
export const URL_PARAMS = {
	/** Load the selected model instead of waiting for the first message. */
	LOAD: 'load',
	/** Model to select. */
	MODEL: 'model',
	/** Prompt to send on arrival. */
	QUERY: 'q'
} as const;

export const ROUTES = {
	/** Chat base — for dynamic chat URLs use RouterService. */
	CHAT: '#/chat',
	/** Models manager. */
	MODELS: '#/models',
	/** Search — mobile-only full-page conversation search. */
	SEARCH: '#/search',
	/** Settings - a page on a phone, a dialog elsewhere. */
	SETTINGS: '#/settings',
	/** Root — start of the app. */
	START: '#/'
} as const;

/** SvelteKit route id of the models manager, as `page.route.id` reports it. */
export const MODELS_ROUTE_ID = '/models';

/** SvelteKit route id of the settings page, as `page.route.id` reports it. */
export const SETTINGS_ROUTE_ID = '/settings';
