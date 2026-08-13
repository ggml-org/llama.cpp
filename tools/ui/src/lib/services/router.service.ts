import { ROUTES } from '$lib/constants';

export class RouterService {
	static chat(id: string): string {
		return `${ROUTES.CHAT}/${id}`;
	}

	static settings(section: string): string {
		return `${ROUTES.SETTINGS}/${section}`;
	}
}
