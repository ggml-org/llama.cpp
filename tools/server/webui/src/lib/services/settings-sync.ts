/**
 * Settings Synchronization Service
 *
 * Coordinates between server store and settings store to sync parameter defaults.
 * This module avoids circular dependencies by importing both stores and handling
 * the coordination logic in a separate service.
 */

import { browser } from '$app/environment';
import { serverStore } from '$lib/stores/server.svelte';
import { settingsStore } from '$lib/stores/settings.svelte';

/**
 * Initialize settings synchronization when server properties are loaded
 * This should be called once during app initialization
 */
export function initializeSettingsSync(): void {
	if (!browser) return;

	// Set up reactive effect to sync settings when server props change
	$effect(() => {
		const serverProps = serverStore.serverProps;

		if (serverProps?.default_generation_settings?.params) {
			console.log('Server props loaded, syncing settings with defaults');
			settingsStore.syncWithServerDefaults();
		}
	});
}

/**
 * Get server default parameters (convenience function)
 */
export function getServerDefaultParams() {
	return serverStore.serverDefaultParams;
}
