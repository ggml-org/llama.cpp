import { beforeNavigate } from '$app/navigation';
import { page } from '$app/state';
import { ROUTES } from '$lib/constants';
import { settingsReferrer } from '$lib/stores';

export interface ChatSettings {
	reset: () => void;
}

export function useSettingsNavigation() {
	const subroute = $state({
		activePanel: 'chat' as 'chat' | 'settings' | 'mcp',
		chatSettingsRef: undefined as ChatSettings | undefined
	});
	const isSettingsRoute = $derived(!!page.route.id?.startsWith('/settings'));

	beforeNavigate(({ from, to }) => {
		if (to?.route?.id?.startsWith('/settings') && !from?.route?.id?.startsWith('/settings')) {
			settingsReferrer.url = window.location.hash || ROUTES.START;
		}
	});

	$effect(() => {
		if (subroute.activePanel === 'settings' && subroute.chatSettingsRef) {
			subroute.chatSettingsRef.reset();
		}
	});

	// Return to chat when navigating to a new route
	$effect(() => {
		void page.url;

		subroute.activePanel = 'chat';
	});

	return {
		get isSettingsRoute() {
			return isSettingsRoute;
		},

		get panel() {
			return subroute;
		}
	};
}
