import type { PageLoad } from './$types';
import { loadAppRoute } from '#lib/stores/init.ts';

export const load: PageLoad = ({ fetch }) => loadAppRoute(fetch);
