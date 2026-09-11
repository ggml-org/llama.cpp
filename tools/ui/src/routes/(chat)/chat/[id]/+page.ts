import type { PageLoad } from './$types';
import { initStores } from '$lib/stores/init';
import { validateApiKey } from '$lib/utils';

export const load: PageLoad = async ({ fetch }) => {
	// loads run before the root layout script, so the stored API key reaches
	// the probe only once the settings store has read localStorage
	await initStores();
	await validateApiKey(fetch);
};
