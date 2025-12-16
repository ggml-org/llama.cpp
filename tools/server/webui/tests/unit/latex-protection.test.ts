/* eslint-disable no-irregular-whitespace */
import { describe, it, expect, test } from 'vitest';
import { maskInlineLaTeX, preprocessLaTeX } from '$lib/utils/latex-protection';

describe('maskInlineLaTeX', () => {
	it('should protect LaTeX $x + y$ but not money $3.99', () => {
		const latexExpressions: string[] = [];
		const input = 'I have $10, $3.99 and $x + y$ and $100x$. The amount is $2,000.';
		const output = maskInlineLaTeX(input, latexExpressions);

		expect(output).toBe('I have $10, $3.99 and <<LATEX_0>> and <<LATEX_1>>. The amount is $2,000.');
		expect(latexExpressions).toEqual(['$x + y$', '$100x$']);
	});

	it('should ignore money like $5 and $12.99', () => {
		const latexExpressions: string[] = [];
		const input = 'Prices are $12.99 and $5. Tax?';
		const output = maskInlineLaTeX(input, latexExpressions);

		expect(output).toBe('Prices are $12.99 and $5. Tax?');
		expect(latexExpressions).toEqual([]);
	});

	it('should protect inline math $a^2 + b^2$ even after text', () => {
		const latexExpressions: string[] = [];
		const input = 'Pythagorean: $a^2 + b^2 = c^2$.';
		const output = maskInlineLaTeX(input, latexExpressions);

		expect(output).toBe('Pythagorean: <<LATEX_0>>.');
		expect(latexExpressions).toEqual(['$a^2 + b^2 = c^2$']);
	});

	it('should not protect math that has letter after closing $ (e.g. units)', () => {
		const latexExpressions: string[] = [];
		const input = 'The cost is $99 and change.';
		const output = maskInlineLaTeX(input, latexExpressions);

		expect(output).toBe('The cost is $99 and change.');
		expect(latexExpressions).toEqual([]);
	});

	it('should allow $x$ followed by punctuation', () => {
		const latexExpressions: string[] = [];
		const input = 'We know $x$, right?';
		const output = maskInlineLaTeX(input, latexExpressions);

		expect(output).toBe('We know <<LATEX_0>>, right?');
		expect(latexExpressions).toEqual(['$x$']);
	});

	it('should work across multiple lines', () => {
		const latexExpressions: string[] = [];
		const input = `Emma buys cupcakes for $3 each.\nHow much is $x + y$?`;
		const output = maskInlineLaTeX(input, latexExpressions);

		expect(output).toBe(`Emma buys cupcakes for $3 each.\nHow much is <<LATEX_0>>?`);
		expect(latexExpressions).toEqual(['$x + y$']);
	});

	it('should not protect $100 but protect $matrix$', () => {
		const latexExpressions: string[] = [];
		const input = '$100 and $\\mathrm{GL}_2(\\mathbb{F}_7)$ are different.';
		const output = maskInlineLaTeX(input, latexExpressions);

		expect(output).toBe('$100 and <<LATEX_0>> are different.');
		expect(latexExpressions).toEqual(['$\\mathrm{GL}_2(\\mathbb{F}_7)$']);
	});

	it('should skip if $ is followed by digit and alphanumeric after close (money)', () => {
		const latexExpressions: string[] = [];
		const input = 'I paid $5 quickly.';
		const output = maskInlineLaTeX(input, latexExpressions);

		expect(output).toBe('I paid $5 quickly.');
		expect(latexExpressions).toEqual([]);
	});

	it('should protect LaTeX even with special chars inside', () => {
		const latexExpressions: string[] = [];
		const input = 'Consider $\\alpha_1 + \\beta_2$ now.';
		const output = maskInlineLaTeX(input, latexExpressions);

		expect(output).toBe('Consider <<LATEX_0>> now.');
		expect(latexExpressions).toEqual(['$\\alpha_1 + \\beta_2$']);
	});

	it('short text', () => {
		const latexExpressions: string[] = ['$0$'];
		const input = '$a$\n$a$ and $b$';
		const output = maskInlineLaTeX(input, latexExpressions);

		expect(output).toBe('<<LATEX_1>>\n<<LATEX_2>> and <<LATEX_3>>');
		expect(latexExpressions).toEqual(['$0$', '$a$', '$a$', '$b$']);
	});

	it('empty text', () => {
		const latexExpressions: string[] = [];
		const input = '$\n$$\n';
		const output = maskInlineLaTeX(input, latexExpressions);

		expect(output).toBe('$\n$$\n');
		expect(latexExpressions).toEqual([]);
	});

	it('LaTeX-spacer preceded by backslash', () => {
		const latexExpressions: string[] = [];
		const input = `\\[
\\boxed{
\\begin{aligned}
N_{\\text{att}}^{\\text{(MHA)}} &=
h \\bigl[\\, d_{\\text{model}}\\;d_{k} + d_{\\text{model}}\\;d_{v}\\, \\bigr]   && (\\text{Q,K,V の重み})\\\\
&\\quad+ h(d_{k}+d_{k}+d_{v})                                          && (\\text{バイアス Q,K,V）}\\\\[4pt]
&\\quad+ (h d_{v})\\, d_{\\text{model}}                                 && (\\text{出力射影 }W^{O})\\\\
&\\quad+ d_{\\text{model}}                                            && (\\text{バイアス }b^{O})
\\end{aligned}}
\\]`;
		const output = maskInlineLaTeX(input, latexExpressions);

		expect(output).toBe(input);
		expect(latexExpressions).toEqual([]);
	});
});

describe('preprocessLaTeX', () => {
	test('converts inline \\( ... \\) to $...$', () => {
		const input =
			'\\( \\mathrm{GL}_2(\\mathbb{F}_7) \\): Group of invertible matrices with entries in \\(\\mathbb{F}_7\\).';
		const output = preprocessLaTeX(input);
		expect(output).toBe(
			'$ \\mathrm{GL}_2(\\mathbb{F}_7) $: Group of invertible matrices with entries in $\\mathbb{F}_7$.'
		);
	});

	test("don't inline \\\\( ... \\) to $...$", () => {
		const input =
			'Chapter 20 of The TeXbook, in source "Definitions\\\\(also called Macros)", containst the formula \\((x_1,\\ldots,x_n)\\).';
		const output = preprocessLaTeX(input);
		expect(output).toBe(
			'Chapter 20 of The TeXbook, in source "Definitions\\\\(also called Macros)", containst the formula $(x_1,\\ldots,x_n)$.'
		);
	});

	test('preserves display math \\[ ... \\] and protects adjacent text', () => {
		const input = `Some kernel of \\(\\mathrm{SL}_2(\\mathbb{F}_7)\\):
  \\[
  \\left\\{ \\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}, \\begin{pmatrix} -1 & 0 \\\\ 0 & -1 \\end{pmatrix} \\right\\} = \\{\\pm I\\}
  \\]`;
		const output = preprocessLaTeX(input);

		expect(output).toBe(`Some kernel of $\\mathrm{SL}_2(\\mathbb{F}_7)$:
  $$
  \\left\\{ \\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}, \\begin{pmatrix} -1 & 0 \\\\ 0 & -1 \\end{pmatrix} \\right\\} = \\{\\pm I\\}
  $$`);
	});

	test('handles standalone display math equation', () => {
		const input = `Algebra:
\\[
x = \\frac{-b \\pm \\sqrt{\\,b^{2}-4ac\\,}}{2a}
\\]`;
		const output = preprocessLaTeX(input);

		expect(output).toBe(`Algebra:
$$
x = \\frac{-b \\pm \\sqrt{\\,b^{2}-4ac\\,}}{2a}
$$`);
	});

	test('does not interpret currency values as LaTeX', () => {
		const input = 'I have $10, $3.99 and $x + y$ and $100x$. The amount is $2,000.';
		const output = preprocessLaTeX(input);

		expect(output).toBe('I have \\$10, \\$3.99 and $x + y$ and $100x$. The amount is \\$2,000.');
	});

	test('ignores dollar signs followed by digits (money), but keeps valid math $x + y$', () => {
		const input = 'I have $10, $3.99 and $x + y$ and $100x$. The amount is $2,000.';
		const output = preprocessLaTeX(input);

		expect(output).toBe('I have \\$10, \\$3.99 and $x + y$ and $100x$. The amount is \\$2,000.');
	});

	test('handles real-world word problems with amounts and no math delimiters', () => {
		const input =
			'Emma buys 2 cupcakes for $3 each and 1 cookie for $1.50. How much money does she spend in total?';
		const output = preprocessLaTeX(input);

		expect(output).toBe(
			'Emma buys 2 cupcakes for \\$3 each and 1 cookie for \\$1.50. How much money does she spend in total?'
		);
	});

	test('handles decimal amounts in word problem correctly', () => {
		const input =
			'Maria has $20. She buys a notebook for $4.75 and a pack of pencils for $3.25. How much change does she receive?';
		const output = preprocessLaTeX(input);

		expect(output).toBe(
			'Maria has \\$20. She buys a notebook for \\$4.75 and a pack of pencils for \\$3.25. How much change does she receive?'
		);
	});

	test('preserves display math with surrounding non-ASCII text', () => {
		const input = `1 kg の質量は
  \\[
  E = (1\\ \\text{kg}) \\times (3.0 \\times 10^8\\ \\text{m/s})^2 \\approx 9.0 \\times 10^{16}\\ \\text{J}
  \\]
  というエネルギーに相当します。これは約 21 百万トンの TNT が爆発したときのエネルギーに匹敵します。`;
		const output = preprocessLaTeX(input);

		expect(output).toBe(
			`1 kg の質量は
  $$
  E = (1\\ \\text{kg}) \\times (3.0 \\times 10^8\\ \\text{m/s})^2 \\approx 9.0 \\times 10^{16}\\ \\text{J}
  $$
  というエネルギーに相当します。これは約 21 百万トンの TNT が爆発したときのエネルギーに匹敵します。`
		);
	});

	test('LaTeX-spacer preceded by backslash', () => {
		const input = `\\[
\\boxed{
\\begin{aligned}
N_{\\text{att}}^{\\text{(MHA)}} &=
h \\bigl[\\, d_{\\text{model}}\\;d_{k} + d_{\\text{model}}\\;d_{v}\\, \\bigr]   && (\\text{Q,K,V の重み})\\\\
&\\quad+ h(d_{k}+d_{k}+d_{v})                                          && (\\text{バイアス Q,K,V）}\\\\[4pt]
&\\quad+ (h d_{v})\\, d_{\\text{model}}                                 && (\\text{出力射影 }W^{O})\\\\
&\\quad+ d_{\\text{model}}                                            && (\\text{バイアス }b^{O})
\\end{aligned}}
\\]`;
		const output = preprocessLaTeX(input);
		expect(output).toBe(
			`$$
\\boxed{
\\begin{aligned}
N_{\\text{att}}^{\\text{(MHA)}} &=
h \\bigl[\\, d_{\\text{model}}\\;d_{k} + d_{\\text{model}}\\;d_{v}\\, \\bigr]   && (\\text{Q,K,V の重み})\\\\
&\\quad+ h(d_{k}+d_{k}+d_{v})                                          && (\\text{バイアス Q,K,V）}\\\\[4pt]
&\\quad+ (h d_{v})\\, d_{\\text{model}}                                 && (\\text{出力射影 }W^{O})\\\\
&\\quad+ d_{\\text{model}}                                            && (\\text{バイアス }b^{O})
\\end{aligned}}
$$`
		);
	});

	test('converts \\[ ... \\] even when preceded by text without space', () => {
		const input = 'Some line ...\nAlgebra: \\[x = \\frac{-b \\pm \\sqrt{\\,b^{2}-4ac\\,}}{2a}\\]';
		const output = preprocessLaTeX(input);

		expect(output).toBe(
			'Some line ...\nAlgebra: \n$$x = \\frac{-b \\pm \\sqrt{\\,b^{2}-4ac\\,}}{2a}$$\n'
		);
	});

	test('converts \\[ ... \\] in table-cells', () => {
		const input = `| ID | Expression |\n| #1 | \\[
			x = \\frac{-b \\pm \\sqrt{\\,b^{2}-4ac\\,}}{2a}
\\] |`;
		const output = preprocessLaTeX(input);

		expect(output).toBe(
			'| ID | Expression |\n| #1 | $x = \\frac{-b \\pm \\sqrt{\\,b^{2}-4ac\\,}}{2a}$ |'
		);
	});

	test('escapes isolated $ before digits ($5 → \\$5), but not valid math', () => {
		const input = 'This costs $5 and this is math $x^2$. $100 is money.';
		const output = preprocessLaTeX(input);

		expect(output).toBe('This costs \\$5 and this is math $x^2$. \\$100 is money.');
		// Note: Since $x^2$ is detected as valid LaTeX, it's preserved.
		// $5 becomes \$5 only *after* real math is masked — but here it's correct because the masking logic avoids treating $5 as math.
	});

	test('display with LaTeX-line-breaks', () => {
		const input = String.raw`- Algebraic topology, Homotopy Groups of $\mathbb{S}^3$:
$$\pi_n(\mathbb{S}^3) = \begin{cases}
\mathbb{Z} & n = 3 \\
0 & n > 3, n \neq 4 \\
\mathbb{Z}_2 & n = 4 \\
\end{cases}$$`;
		const output = preprocessLaTeX(input);
		// If the formula contains '\\' the $$-delimiters should be in their own line.
		expect(output).toBe(`- Algebraic topology, Homotopy Groups of $\\mathbb{S}^3$:
$$\n\\pi_n(\\mathbb{S}^3) = \\begin{cases}
\\mathbb{Z} & n = 3 \\\\
0 & n > 3, n \\neq 4 \\\\
\\mathbb{Z}_2 & n = 4 \\\\
\\end{cases}\n$$`);
	});

	test('handles mhchem notation safely if present', () => {
		const input = 'Chemical reaction: \\( \\ce{H2O} \\) and $\\ce{CO2}$';
		const output = preprocessLaTeX(input);

		expect(output).toBe('Chemical reaction: $ \\ce{H2O} $ and $\\ce{CO2}$');
	});

	test('preserves code blocks', () => {
		const input = 'Inline code: `sum $total` and block:\n```\ndollar $amount\n```\nEnd.';
		const output = preprocessLaTeX(input);

		expect(output).toBe(input); // Code blocks prevent misinterpretation
	});

	test('preserves backslash parentheses in code blocks (GitHub issue)', () => {
		const input = '```python\nfoo = "\\(bar\\)"\n```';
		const output = preprocessLaTeX(input);

		expect(output).toBe(input); // Code blocks should not have LaTeX conversion applied
	});

	test('preserves backslash brackets in code blocks', () => {
		const input = '```python\nfoo = "\\[bar\\]"\n```';
		const output = preprocessLaTeX(input);

		expect(output).toBe(input); // Code blocks should not have LaTeX conversion applied
	});

	test('preserves backslash parentheses in inline code', () => {
		const input = 'Use `foo = "\\(bar\\)"` in your code.';
		const output = preprocessLaTeX(input);

		expect(output).toBe(input);
	});

	test('escape backslash in mchem ce', () => {
		const input = 'mchem ce:\n$\\ce{2H2(g) + O2(g) -> 2H2O(l)}$';
		const output = preprocessLaTeX(input);

		// mhchem-escape would insert a backslash here.
		expect(output).toBe('mchem ce:\n$\\ce{2H2(g) + O2(g) -> 2H2O(l)}$');
	});

	test('escape backslash in mchem pu', () => {
		const input = 'mchem pu:\n$\\pu{-572 kJ mol^{-1}}$';
		const output = preprocessLaTeX(input);

		// mhchem-escape would insert a backslash here.
		expect(output).toBe('mchem pu:\n$\\pu{-572 kJ mol^{-1}}$');
	});

	test('LaTeX in blockquotes with display math', () => {
		const input =
			'> **Definition (limit):**  \n>  \\[\n>  \\lim_{x\\to a} f(x) = L\n>  \\]\n>  means that as \\(x\\) gets close to \\(a\\).';
		const output = preprocessLaTeX(input);

		// Blockquote markers should be preserved, LaTeX should be converted
		expect(output).toContain('> **Definition (limit):**');
		expect(output).toContain('$$');
		expect(output).toContain('$x$');
		expect(output).not.toContain('\\[');
		expect(output).not.toContain('\\]');
		expect(output).not.toContain('\\(');
		expect(output).not.toContain('\\)');
	});

	test('LaTeX in blockquotes with inline math', () => {
		const input =
			"> The derivative \\(f'(x)\\) at point \\(x=a\\) measures slope.\n> Formula: \\(f'(a)=\\lim_{h\\to 0}\\frac{f(a+h)-f(a)}{h}\\)";
		const output = preprocessLaTeX(input);

		// Blockquote markers should be preserved, inline LaTeX converted to $...$
		expect(output).toContain("> The derivative $f'(x)$ at point $x=a$ measures slope.");
		expect(output).toContain("> Formula: $f'(a)=\\lim_{h\\to 0}\\frac{f(a+h)-f(a)}{h}$");
	});

	test('Mixed content with blockquotes and regular text', () => {
		const input =
			'Regular text with \\(x^2\\).\n\n> Quote with \\(y^2\\).\n\nMore text with \\(z^2\\).';
		const output = preprocessLaTeX(input);

		// All LaTeX should be converted, blockquote markers preserved
		expect(output).toBe('Regular text with $x^2$.\n\n> Quote with $y^2$.\n\nMore text with $z^2$.');
	});
});
