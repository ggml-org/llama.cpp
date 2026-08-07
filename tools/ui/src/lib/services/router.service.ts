import { ROUTES } from '$lib/constants/routes';

export class RouterService {
	static chat(id: string): string {
		return `${ROUTES.CHAT}/${id}`;
	}

	static settings(section: string): string {
		return `${ROUTES.SETTINGS}/${section}`;
	}

	static model(modelId: string): string {
		return ROUTES.MANAGE_MODEL.replace('[modelId]', modelId);
	}

	/**
	 * Build a Model Hub URL from a model id that may carry a `:tag` suffix
	 * (the form used by the server in /v1/models — e.g.
	 * `ggml-org/gemma-3-4b-it-GGUF:Q4_K_M`). The Model Hub detail page lists
	 * every available quantization for the repo, so we always drop the tag and
	 * land on the base repo page.
	 *
	 * @param modelId - Model id, optionally suffixed with `:tag`
	 * @returns Model Hub URL pointing at the repo (tag stripped)
	 */
	static fromModelId(modelId: string): string {
		const stripped = modelId.split(':')[0] ?? modelId;
		return ROUTES.MANAGE_MODEL.replace('[modelId]', stripped);
	}
}
