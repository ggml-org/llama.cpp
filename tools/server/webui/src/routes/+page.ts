import type { PageLoad } from './$types';
import { validateApiKey } from '$lib/utils/api-key-validation';

export const load: PageLoad = async ({ fetch }) => {
	await validateApiKey(fetch);
};
