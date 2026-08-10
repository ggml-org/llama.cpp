// For more info, see https://github.com/storybookjs/eslint-plugin-storybook#configuration-flat-config-format
import svelteConfig from './svelte.config.js';
import { includeIgnoreFile } from '@eslint/compat';
import js from '@eslint/js';
import prettier from 'eslint-config-prettier';
import perfectionist from 'eslint-plugin-perfectionist';
import simpleImportSort from 'eslint-plugin-simple-import-sort';
import storybook from 'eslint-plugin-storybook';
import svelte from 'eslint-plugin-svelte';
import globals from 'globals';
import { fileURLToPath } from 'node:url';
import ts from 'typescript-eslint';

const gitignorePath = fileURLToPath(new URL('./.gitignore', import.meta.url));

export default ts.config(
	includeIgnoreFile(gitignorePath),
	js.configs.recommended,
	...ts.configs.recommended,
	...svelte.configs.recommended,
	prettier,
	...svelte.configs.prettier,
	{
		languageOptions: { globals: { ...globals.browser, ...globals.node } },
		plugins: { perfectionist, 'simple-import-sort': simpleImportSort },
		rules: {
			// Snippet bodies often ignore one or more of the parent's params
			// (e.g. `{#snippet children(_meta, ctx)}` when only ctx is read).
			'@typescript-eslint/no-unused-vars': [
				'error',
				{ argsIgnorePattern: '^_', varsIgnorePattern: '^_' }
			],
			// Enforce empty line at end of file
			'eol-last': 'error',
			// typescript-eslint strongly recommend that you do not use the no-undef lint rule on TypeScript projects.
			// see: https://typescript-eslint.io/troubleshooting/faqs/eslint/#i-get-errors-from-the-no-undef-rule-about-global-variables-not-being-defined-even-though-there-are-no-typescript-errors
			'no-undef': 'off',

			'padding-line-between-statements': [
				'error',
				// Blank line between function/class declarations.
				{ blankLine: 'always', next: ['function', 'class'], prev: ['function', 'class'] },
				// Blank line around if blocks (if/else and else if stay one statement).
				{ blankLine: 'always', next: '*', prev: 'if' },
				{ blankLine: 'always', next: 'if', prev: '*' },
				// Blank line after the last declaration in a group. Because the 'never'
				// rules below are scoped per declaration kind, a const group and a let
				// group get separated by a blank line, while same-kind declarations stay
				// together.
				{ blankLine: 'always', next: '*', prev: ['const', 'let', 'var'] },
				// No blank line between consecutive declarations of the same kind (kept
				// last so each takes precedence over the always rule above for matching
				// declaration pairs).
				{ blankLine: 'never', next: 'const', prev: 'const' },
				{ blankLine: 'never', next: 'let', prev: 'let' },
				{ blankLine: 'never', next: 'var', prev: 'var' },
				// Blank line before a statement that follows another statement in the block
				// (works for return/throw/break/continue). A blank line for a terminal
				// statement that opens a block body can't be enforced here: Prettier removes
				// the leading blank line of a block, so the two formatters would fight.
				{ blankLine: 'always', next: ['return', 'throw', 'break', 'continue'], prev: '*' }
			],

			'perfectionist/sort-objects': ['error', { type: 'natural' }],

			// Alphabetical order for variable declarations and object keys
			'perfectionist/sort-variable-declarations': ['error', { type: 'natural' }],

			// Sort imports alphabetically by module path, and sort named members within
			// each statement. A single catch-all group keeps the list flat (no blank-line
			// grouping); Prettier normalizes comma spacing afterwards.
			'simple-import-sort/imports': ['error', { groups: [['.*']] }],
			'svelte/no-at-html-tags': 'off',

			// This app uses hash-based routing (#/) where resolve() from $app/paths does not apply
			'svelte/no-navigation-without-resolve': 'off'
		}
	},
	{
		files: ['**/*.svelte', '**/*.svelte.ts', '**/*.svelte.js'],
		languageOptions: {
			parserOptions: {
				extraFileExtensions: ['.svelte'],
				parser: ts.parser,
				projectService: true,
				svelteConfig
			}
		}
	},
	{
		// Exclude generated build output and Storybook files from ESLint
		ignores: [
			'dist/**',
			'build/**',
			'.svelte-kit/**',
			'test-results/**',
			'.storybook/**/*',
			'src/lib/services/sandbox-worker.js',
			'src/lib/vendors/**'
		]
	},
	storybook.configs['flat/recommended']
);
