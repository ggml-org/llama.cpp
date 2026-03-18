import { getJsonHeaders } from '$lib/utils';

/**
 * Tokenizes the provided text using the server's tokenizer.
 *
 * @param content - The text content to tokenize
 * @param model - Optional model name to use for tokenization (required in router mode)
 * @param signal - Optional AbortSignal
 * @returns {Promise<number[]>} Promise that resolves to an array of token IDs
 */
export async function tokenize(
	content: string,
	model?: string,
	signal?: AbortSignal
): Promise<number[]> {
	try {
		const body: { content: string; model?: string } = { content };
		if (model) {
			body.model = model;
		}

		const response = await fetch('./tokenize', {
			method: 'POST',
			headers: getJsonHeaders(),
			body: JSON.stringify(body),
			signal
		});

		if (!response.ok) {
			throw new Error(`Tokenize failed: ${response.statusText}`);
		}

		const data = await response.json();
		return data.tokens;
	} catch (error) {
		console.error('Tokenize error:', error);
		return [];
	}
}
