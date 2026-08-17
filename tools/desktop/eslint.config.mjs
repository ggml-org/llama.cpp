import * as shared from '../../scripts/codestyle/eslint.js';
import { includeIgnoreFile } from '@eslint/compat';
import js from '@eslint/js';
import prettier from 'eslint-config-prettier';
import perfectionist from 'eslint-plugin-perfectionist';
import simpleImportSort from 'eslint-plugin-simple-import-sort';
import globals from 'globals';
import { fileURLToPath } from 'node:url';
import ts from 'typescript-eslint';

const gitignorePath = fileURLToPath(new URL('./.gitignore', import.meta.url));

export default ts.config(
	includeIgnoreFile(gitignorePath),
	js.configs.recommended,
	...ts.configs.recommended,
	prettier,
	{
		languageOptions: { globals: { ...globals.browser, ...globals.node } },
		plugins: shared.getPlugins(perfectionist, simpleImportSort),
		rules: shared.rules
	},
	{
		// Exclude generated build output from ESLint
		ignores: ['dist/**', 'build/**', '.vite/**', 'out/**']
	}
);
