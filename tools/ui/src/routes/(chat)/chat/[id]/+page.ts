import type { PageLoad } from './$types';
import { hydrateStores } from '$lib/stores/init';
import { validateApiKey } from '$lib/utils';

export const load: PageLoad = async ({ fetch }) => {
	// loads run before the root layout script, so the stored API key reaches
	// the probe only once the settings store has read localStorage; the rest of
	// the startup belongs to the layout, after this load
	await hydrateStores();
	await validateApiKey(fetch);
};
