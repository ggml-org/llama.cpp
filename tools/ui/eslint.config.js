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
// Require a blank line between consecutive class accessors (get/set). The core
// `padding-line-between-statements` rule only handles statements, not class
// members, so this is enforced with a small custom rule.
const blankLineBetweenAccessors = {
	create(context) {
		return {
			MethodDefinition(node) {
				if (node.kind !== 'get' && node.kind !== 'set') return;

				const body = node.parent;

				if (!body || body.type !== 'ClassBody') return;

				const index = body.body.indexOf(node);

				if (index <= 0) return;

				const prev = body.body[index - 1];

				if (prev.type !== 'MethodDefinition' || (prev.kind !== 'get' && prev.kind !== 'set'))
					return;

				if (node.loc.start.line - prev.loc.end.line <= 1) {
					context.report({
						fix(fixer) {
							// Insert after the previous accessor's closing brace so the blank
							// line keeps the current accessor's indentation.
							return fixer.insertTextAfter(prev, '\n');
						},
						message: 'Expected a blank line between class accessors (get/set).',
						node
					});
				}
			}
		};
	},
	meta: {
		docs: { description: 'Require a blank line between consecutive class accessors (get/set).' },
		fixable: 'whitespace',
		schema: [],
		type: 'layout'
	}
};

export default ts.config(
	includeIgnoreFile(gitignorePath),
	js.configs.recommended,
	...ts.configs.recommended,
	...svelte.configs.recommended,
	prettier,
	...svelte.configs.prettier,
	{
		languageOptions: { globals: { ...globals.browser, ...globals.node } },
		plugins: {
			local: { rules: { 'blank-line-between-accessors': blankLineBetweenAccessors } },
			perfectionist,
			'simple-import-sort': simpleImportSort
		},
		rules: {
			// Snippet bodies often ignore one or more of the parent's params
			// (e.g. `{#snippet children(_meta, ctx)}` when only ctx is read).
			'@typescript-eslint/no-unused-vars': [
				'error',
				{ argsIgnorePattern: '^_', varsIgnorePattern: '^_' }
			],

			// Enforce empty line at end of file
			'eol-last': 'error',
			// Enforce a blank line between consecutive get/set accessors
			'local/blank-line-between-accessors': 'error',
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

			// Class member order: public fields -> private fields -> constructor -> getters
			// -> setters -> public methods -> private methods, alphabetical within each.
			// Svelte $derived fields must stay in dependency order (forward references are
			// rejected), so the two stores that rely on that are exempted below.
			'perfectionist/sort-classes': [
				'error',
				{
					customGroups: [
						{ groupName: 'public-field', modifiers: ['public'], selector: 'property' },
						{ groupName: 'private-field', modifiers: ['private'], selector: 'property' },
						{ groupName: 'get-method', selector: 'get-method' },
						{ groupName: 'set-method', selector: 'set-method' },
						{ groupName: 'public-method', modifiers: ['public'], selector: 'method' },
						{ groupName: 'private-method', modifiers: ['private'], selector: 'method' }
					],
					groups: [
						'public-field',
						'private-field',
						'constructor',
						'get-method',
						'set-method',
						'public-method',
						'private-method',
						'unknown'
					],
					type: 'natural',
					// Keep members in dependency order (Svelte rejects forward references in
					// $derived fields), while still sorting the rest alphabetically.
					useExperimentalDependencyDetection: true
				}
			],

			// Alphabetical order for enum members
			'perfectionist/sort-enums': ['error', { type: 'natural' }],

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
