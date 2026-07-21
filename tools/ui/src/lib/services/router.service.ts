import { ROUTES } from '$lib/constants/routes';

export class RouterService {
	static chat(id: string): string {
		return `${ROUTES.CHAT}/${id}`;
	}

	static settings(section: string): string {
		return `${ROUTES.SETTINGS}/${section}`;
	}

	static model(modelId: string): string {
		return ROUTES.MANAGE_MODEL.replace('[modelId]', encodeURIComponent(modelId));
	}
}
