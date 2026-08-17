// For more info, see https://github.com/storybookjs/eslint-plugin-storybook#configuration-flat-config-format
import * as shared from '../../scripts/codestyle/eslint.js';
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
		plugins: shared.getPlugins(perfectionist, simpleImportSort),
		rules: {
			...shared.rules,

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
