/* eslint-disable no-irregular-whitespace */
import { describe, it, expect, test } from 'vitest';
import { maskInlineLaTeX, preprocessLaTeX } from './latex-protection';

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

	test('handles mhchem notation safely if present', () => {
		const input = 'Chemical reaction: \\( \\ce{H2O} \\) and $\\ce{CO2}$';
		const output = preprocessLaTeX(input);

		expect(output).toBe('Chemical reaction: $ \\ce{H2O} $ and $\\\\ce{CO2}$');
		// Note: \\ce{...} remains, but $\\ce{...} → $\\\\ce{...} via escapeMhchem
	});

	test('preserves code blocks', () => {
		const input = 'Inline code: `sum $total` and block:\n```\ndollar $amount\n```\nEnd.';
		const output = preprocessLaTeX(input);

		expect(output).toBe(input); // Code blocks prevent misinterpretation
	});
});
