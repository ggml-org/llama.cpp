import { API_TOKENIZE } from '$lib/constants/api-endpoints';

interface TokenizeResponse {
	tokens: number[];
}

/**
 * Wraps llama-server's POST /tokenize endpoint. Used to count how many tokens
 * the loaded model's tokenizer produces for a given piece of text — for
 * example, to attribute the static cost of enabled MCP tool definitions to
 * the context gauge.
 */
export class TokenizeService {
	/**
	 * Returns the number of tokens the model tokenizer produces for `text`.
	 *
	 * Returns null when tokenization is unavailable (no model loaded, endpoint
	 * unreachable, response shape unexpected). Callers should treat null as
	 * "unknown" and either show a placeholder or fall back to a heuristic.
	 *
	 * @param text - The text to tokenize
	 * @param model - Required in ROUTER mode so the request is forwarded to
	 *   the right child instance. Ignored in single-model mode.
	 */
	static async count(text: string, model?: string | null): Promise<number | null> {
		if (!text) return 0;

		try {
			const url =
				model && model.length > 0
					? `${API_TOKENIZE.BASE}?model=${encodeURIComponent(model)}`
					: API_TOKENIZE.BASE;

			const res = await fetch(url, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ content: text })
			});

			if (!res.ok) return null;

			const data: TokenizeResponse = await res.json();
			if (!Array.isArray(data.tokens)) return null;
			return data.tokens.length;
		} catch {
			return null;
		}
	}
}
