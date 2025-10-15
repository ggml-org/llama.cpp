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

function escapeBrackets(text: string): string {
	const pattern = /(```[\S\s]*?```|`.*?`)|\\\[([\S\s]*?[^\\])\\]|\\\((.*?)\\\)/g;
	return text.replace(
		pattern,
		(
			match: string,
			codeBlock: string | undefined,
			squareBracket: string | undefined,
			roundBracket: string | undefined
		): string => {
			if (codeBlock != null) {
				return codeBlock;
			} else if (squareBracket != null) {
				return `$$${squareBracket}$$`;
			} else if (roundBracket != null) {
				return `$${roundBracket}$`;
			}
			return match;
		}
	);
}

// Escape $\\ce{...} → $\\ce{...} but with proper handling
function escapeMhchem(text: string): string {
	return text.replaceAll('$\\ce{', '$\\\\ce{').replaceAll('$\\pu{', '$\\\\pu{');
}

// See also:
// https://github.com/danny-avila/LibreChat/blob/main/client/src/utils/latex.ts

// Protect code blocks: ```...``` and `...`
const codeBlockRegex = /(```[\s\S]*?```|`[^`\n]+`)/g;

/**
 * Preprocesses markdown content to safely handle LaTeX math expressions while protecting
 * against false positives (e.g., dollar amounts like $5.99) and ensuring proper rendering.
 *
 * This function:
 * - Protects code blocks (```) and inline code (`...`)
 * - Safeguards block and inline LaTeX: \(...\), \[...\], $$...$$, and selective $...$
 * - Escapes standalone dollar signs before numbers (e.g., $5 → \$5) to prevent misinterpretation
 * - Restores protected LaTeX and code blocks after processing
 * - Converts \(...\) → $...$ and \[...\] → $$...$$ for compatibility with math renderers
 * - Applies additional escaping for brackets and mhchem syntax if needed
 *
 * @param content - The raw text (e.g., markdown) that may contain LaTeX or code blocks.
 * @returns The preprocessed string with properly escaped and normalized LaTeX.
 *
 * @example
 * preprocessLaTeX("Price: $10. The equation is \\(x^2\\).")
 * // → "Price: $10. The equation is $x^2$."
 */
export function preprocessLaTeX(content: string): string {
	// Step 1: Protect code blocks
	const codeBlocks: string[] = [];
	content = content.replace(codeBlockRegex, (match) => {
		codeBlocks.push(match);
		return `<<CODE_BLOCK_${codeBlocks.length - 1}>>`;
	});

	// Step 2: Protect existing LaTeX expressions
	const latexExpressions: string[] = [];

	// Match \(...\), \[...\], $$...$$ and protect them
	content = content.replace(/(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\(.*?\\\))/g, (match) => {
		latexExpressions.push(match);
		return `<<LATEX_${latexExpressions.length - 1}>>`;
	});

	// Protect inline $...$ but NOT if it looks like money (e.g., $10, $3.99)
	content = maskInlineLaTeX(content, latexExpressions);

	// Step 3: Escape standalone $ before digits (currency like $5 → \$5)
	// (Now that inline math is protected, this will only escape dollars not already protected)
	content = content.replace(/\$(?=\d)/g, '\\$');

	// Step 4: Restore protected LaTeX expressions (they are valid)
	content = content.replace(/<<LATEX_(\d+)>>/g, (_, index) => {
		return latexExpressions[parseInt(index)];
	});

	// Step 5: Restore code blocks
	content = content.replace(/<<CODE_BLOCK_(\d+)>>/g, (_, index) => {
		return codeBlocks[parseInt(index)];
	});

	// Step 6: Apply additional escaping functions (brackets and mhchem)
	content = escapeBrackets(content);
	if (content.includes('\\ce{') || content.includes('\\pu{')) {
		content = escapeMhchem(content);
	}

	// Final pass: Convert \(...\) → $...$, \[...\] → $$...$$
	content = content
		.replace(/\\\((.+?)\\\)/g, '$$$1$') // inline
		.replace(/\\\[(.+?)\\\]/g, '$$$$1$$'); // display

	return content;
}
