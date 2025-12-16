import {
	CODE_BLOCK_REGEXP,
	LATEX_MATH_AND_CODE_PATTERN,
	LATEX_LINEBREAK_REGEXP,
	MHCHEM_PATTERN_MAP
} from '$lib/constants/latex-protection';

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
	if (!content.includes('$')) {
		return content;
	}
	return content
		.split('\n')
		.map((line) => {
			if (line.indexOf('$') == -1) {
				return line;
			}

			let processedLine = '';
			let currentPosition = 0;

			while (currentPosition < line.length) {
				const openDollarIndex = line.indexOf('$', currentPosition);

				if (openDollarIndex == -1) {
					processedLine += line.slice(currentPosition);
					break;
				}

				// Is there a next $-sign?
				const closeDollarIndex = line.indexOf('$', openDollarIndex + 1);

				if (closeDollarIndex == -1) {
					processedLine += line.slice(currentPosition);
					break;
				}

				const charBeforeOpen = openDollarIndex > 0 ? line[openDollarIndex - 1] : '';
				const charAfterOpen = line[openDollarIndex + 1];
				const charBeforeClose =
					openDollarIndex + 1 < closeDollarIndex ? line[closeDollarIndex - 1] : '';
				const charAfterClose = closeDollarIndex + 1 < line.length ? line[closeDollarIndex + 1] : '';

				let shouldSkipAsNonLatex = false;

				if (closeDollarIndex == currentPosition + 1) {
					// No content
					shouldSkipAsNonLatex = true;
				}

				if (/[A-Za-z0-9_$-]/.test(charBeforeOpen)) {
					// Character, digit, $, _ or - before first '$', no TeX.
					shouldSkipAsNonLatex = true;
				}

				if (
					/[0-9]/.test(charAfterOpen) &&
					(/[A-Za-z0-9_$-]/.test(charAfterClose) || ' ' == charBeforeClose)
				) {
					// First $ seems to belong to an amount.
					shouldSkipAsNonLatex = true;
				}

				if (shouldSkipAsNonLatex) {
					processedLine += line.slice(currentPosition, openDollarIndex + 1);
					currentPosition = openDollarIndex + 1;

					continue;
				}

				// Treat as LaTeX
				processedLine += line.slice(currentPosition, openDollarIndex);
				const latexContent = line.slice(openDollarIndex, closeDollarIndex + 1);
				latexExpressions.push(latexContent);
				processedLine += `<<LATEX_${latexExpressions.length - 1}>>`;
				currentPosition = closeDollarIndex + 1;
			}

			return processedLine;
		})
		.join('\n');
}

function escapeBrackets(text: string): string {
	return text.replace(
		LATEX_MATH_AND_CODE_PATTERN,
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
	return MHCHEM_PATTERN_MAP.reduce((result, [pattern, replacement]) => {
		return result.replace(pattern, replacement);
	}, text);
}

const doEscapeMhchem = false;

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
	// See also:
	// https://github.com/danny-avila/LibreChat/blob/main/client/src/utils/latex.ts

	// Step 0: Temporarily remove blockquote markers (>) to process LaTeX correctly
	// Store the structure so we can restore it later
	const blockquoteMarkers: Map<number, string> = new Map();
	const lines = content.split('\n');
	const processedLines = lines.map((line, index) => {
		const match = line.match(/^(>\s*)/);
		if (match) {
			blockquoteMarkers.set(index, match[1]);
			return line.slice(match[1].length);
		}
		return line;
	});
	content = processedLines.join('\n');

	// Step 1: Protect code blocks
	const codeBlocks: string[] = [];

	content = content.replace(CODE_BLOCK_REGEXP, (match) => {
		codeBlocks.push(match);

		return `<<CODE_BLOCK_${codeBlocks.length - 1}>>`;
	});

	// Step 2: Protect existing LaTeX expressions
	const latexExpressions: string[] = [];

	// Match \S...\[...\] and protect them and insert a line-break.
	content = content.replace(/([\S].*?)\\\[([\s\S]*?)\\\](.*)/g, (match, group1, group2, group3) => {
		// Check if there are characters following the formula (display-formula in a table-cell?)
		if (group1.endsWith('\\')) {
			return match; // Backslash before \[, do nothing.
		}
		const hasSuffix = /\S/.test(group3);
		let optBreak;

		if (hasSuffix) {
			latexExpressions.push(`\\(${group2.trim()}\\)`); // Convert into inline.
			optBreak = '';
		} else {
			latexExpressions.push(`\\[${group2}\\]`);
			optBreak = '\n';
		}

		return `${group1}${optBreak}<<LATEX_${latexExpressions.length - 1}>>${optBreak}${group3}`;
	});

	// Match \(...\), \[...\], $$...$$ and protect them
	content = content.replace(
		/(\$\$[\s\S]*?\$\$|(?<!\\)\\\[[\s\S]*?\\\]|(?<!\\)\\\(.*?\\\))/g,
		(match) => {
			latexExpressions.push(match);

			return `<<LATEX_${latexExpressions.length - 1}>>`;
		}
	);

	// Protect inline $...$ but NOT if it looks like money (e.g., $10, $3.99)
	content = maskInlineLaTeX(content, latexExpressions);

	// Step 3: Escape standalone $ before digits (currency like $5 → \$5)
	// (Now that inline math is protected, this will only escape dollars not already protected)
	content = content.replace(/\$(?=\d)/g, '\\$');

	// Step 4: Restore protected LaTeX expressions (they are valid)
	content = content.replace(/<<LATEX_(\d+)>>/g, (_, index) => {
		let expr = latexExpressions[parseInt(index)];
		const match = expr.match(LATEX_LINEBREAK_REGEXP);
		if (match) {
			// Katex: The $$-delimiters should be in their own line
			// if there are \\-line-breaks.
			const formula = match[1];
			const prefix = formula.startsWith('\n') ? '' : '\n';
			const suffix = formula.endsWith('\n') ? '' : '\n';
			expr = '$$' + prefix + formula + suffix + '$$';
		}
		return expr;
	});

	// Step 5: Apply additional escaping functions (brackets and mhchem)
	// This must happen BEFORE restoring code blocks to avoid affecting code content
	content = escapeBrackets(content);

	if (doEscapeMhchem && (content.includes('\\ce{') || content.includes('\\pu{'))) {
		content = escapeMhchem(content);
	}

	// Step 6: Convert remaining \(...\) → $...$, \[...\] → $$...$$
	// This must happen BEFORE restoring code blocks to avoid affecting code content
	content = content
		// Using the look‑behind pattern `(?<!\\)` we skip matches
		// that are preceded by a backslash, e.g.
		// `Definitions\\(also called macros)` (title of chapter 20 in The TeXbook).
		.replace(/(?<!\\)\\\((.+?)\\\)/g, '$$$1$') // inline
		.replace(
			// Using the look‑behind pattern `(?<!\\)` we skip matches
			// that are preceded by a backslash, e.g. `\\[4pt]`.
			/(?<!\\)\\\[([\s\S]*?)\\\]/g, // display, see also PR #16599
			(_, content: string) => {
				return `$$${content}$$`;
			}
		);

	// Step 7: Restore code blocks
	// This happens AFTER all LaTeX conversions to preserve code content
	content = content.replace(/<<CODE_BLOCK_(\d+)>>/g, (_, index) => {
		return codeBlocks[parseInt(index)];
	});

	// Step 8: Restore blockquote markers
	if (blockquoteMarkers.size > 0) {
		const finalLines = content.split('\n');
		const restoredLines = finalLines.map((line, index) => {
			const marker = blockquoteMarkers.get(index);
			return marker ? marker + line : line;
		});
		content = restoredLines.join('\n');
	}

	return content;
}
