import { page } from '$app/state';
import { beforeNavigate } from '$app/navigation';
import { settingsReferrer } from '$lib/stores/settings-referrer.svelte';

export interface ChatSettings {
	reset: () => void;
}

export function useSettingsNavigation() {
	const subroute = $state({
		activePanel: 'chat' as 'chat' | 'settings' | 'mcp',
		chatSettingsRef: undefined as ChatSettings | undefined
	});

	const isSettingsRoute = $derived(!!page.route.id?.startsWith('/settings'));

	beforeNavigate(({ to, from }) => {
		if (
			to?.route?.id?.startsWith('/settings/chat') &&
			!from?.route?.id?.startsWith('/settings/chat')
		) {
			settingsReferrer.url = window.location.hash || '#/';
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
		get panel() {
			return subroute;
		},

		get isSettingsRoute() {
			return isSettingsRoute;
		}
	};
}
