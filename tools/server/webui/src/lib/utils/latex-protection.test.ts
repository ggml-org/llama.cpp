import { describe, it, expect } from 'vitest';
import { maskInlineLaTeX } from './latex-protection';

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
