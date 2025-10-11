/**
 * Replaces inline LaTeX expressions enclosed in `$...$` with placeholders, avoiding dollar signs
 * that appear to be part of monetary values or identifiers.
 *
 * This function processes the input line by line and skips `$` sequences that are likely
 * part of money amounts (e.g., `$5`, `$100.99`) or code-like tokens (e.g., `var$`, `$var`).
 * Valid LaTeX inline math is replaced with a placeholder like `<<LATEX_0>>`, and the
 * actual LaTeX content is stored in the provided `latexExpressions` array.
 *
 * @param content - The input text potentially containing LaTeX expressions.
 * @param latexExpressions - An array used to collect extracted LaTeX expressions.
 * @returns The processed string with LaTeX replaced by placeholders.
 */
export function maskInlineLaTeX(content: string, latexExpressions: string[]): string {
	if (content.indexOf('$') == -1) {
		return content;
	}
	return content
		.split('\n')
		.map((line) => {
			if (line.indexOf('$') == -1) {
				return line;
			}
			let result = '';
			let index = 0;
			while (index < line.length) {
				const openIndex = line.indexOf('$', index);
				if (openIndex == -1) {
					result += line.slice(index);
					break;
				}

				// Is there a next $-sign?
				const nextIndex = line.indexOf('$', openIndex + 1);
				if (nextIndex == -1) {
					result += line.slice(index);
					break;
				}

				const beforeOpenChar = openIndex > 0 ? line[openIndex - 1] : '';
				const afterOpenChar = line[openIndex + 1];
				const beforeCloseChar = openIndex + 1 < nextIndex ? line[nextIndex - 1] : '';
				const afterCloseChar = nextIndex + 1 < line.length ? line[nextIndex + 1] : '';
				let cont = false;
				if (nextIndex == index + 1) {
					// no content
					cont = true;
				}
				if (/[A-Za-z0-9_$-]/.test(beforeOpenChar)) {
					// character, digit, $, _ or - before first '$', no TeX.
					cont = true;
				}
				if (
					/[0-9]/.test(afterOpenChar) &&
					(/[A-Za-z0-9_$-]/.test(afterCloseChar) || ' ' == beforeCloseChar)
				) {
					// First $ seems to belong to an amount.
					cont = true;
				}
				if (cont) {
					result += line.slice(index, openIndex + 1);
					index = openIndex + 1;
					continue;
				}

				// Treat as LaTeX
				result += line.slice(index, openIndex);
				const latexContent = line.slice(openIndex, nextIndex + 1);
				latexExpressions.push(latexContent);
				result += `<<LATEX_${latexExpressions.length - 1}>>`;
				index = nextIndex + 1;
			}
			return result;
		})
		.join('\n');
}
