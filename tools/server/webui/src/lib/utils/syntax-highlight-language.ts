/**
 * Maps file extensions to highlight.js language identifiers
 */
export function getLanguageFromFilename(filename: string): string {
	const extension = filename.toLowerCase().substring(filename.lastIndexOf('.'));

	switch (extension) {
		// JavaScript / TypeScript
		case '.js':
		case '.mjs':
		case '.cjs':
			return 'javascript';
		case '.ts':
		case '.mts':
		case '.cts':
			return 'typescript';
		case '.jsx':
			return 'javascript';
		case '.tsx':
			return 'typescript';

		// Web
		case '.html':
		case '.htm':
			return 'html';
		case '.css':
			return 'css';
		case '.scss':
			return 'scss';
		case '.less':
			return 'less';
		case '.vue':
			return 'html';
		case '.svelte':
			return 'html';

		// Data formats
		case '.json':
			return 'json';
		case '.xml':
			return 'xml';
		case '.yaml':
		case '.yml':
			return 'yaml';
		case '.toml':
			return 'ini';
		case '.csv':
			return 'plaintext';

		// Programming languages
		case '.py':
			return 'python';
		case '.java':
			return 'java';
		case '.kt':
		case '.kts':
			return 'kotlin';
		case '.scala':
			return 'scala';
		case '.cpp':
		case '.cc':
		case '.cxx':
		case '.c++':
			return 'cpp';
		case '.c':
			return 'c';
		case '.h':
		case '.hpp':
			return 'cpp';
		case '.cs':
			return 'csharp';
		case '.go':
			return 'go';
		case '.rs':
			return 'rust';
		case '.rb':
			return 'ruby';
		case '.php':
			return 'php';
		case '.swift':
			return 'swift';
		case '.dart':
			return 'dart';
		case '.r':
			return 'r';
		case '.lua':
			return 'lua';
		case '.pl':
		case '.pm':
			return 'perl';

		// Shell
		case '.sh':
		case '.bash':
		case '.zsh':
			return 'bash';
		case '.bat':
		case '.cmd':
			return 'dos';
		case '.ps1':
			return 'powershell';

		// Database
		case '.sql':
			return 'sql';

		// Markup / Documentation
		case '.md':
		case '.markdown':
			return 'markdown';
		case '.tex':
		case '.latex':
			return 'latex';
		case '.adoc':
		case '.asciidoc':
			return 'asciidoc';

		// Config
		case '.ini':
		case '.cfg':
		case '.conf':
			return 'ini';
		case '.dockerfile':
			return 'dockerfile';
		case '.nginx':
			return 'nginx';

		// Other
		case '.graphql':
		case '.gql':
			return 'graphql';
		case '.proto':
			return 'protobuf';
		case '.diff':
		case '.patch':
			return 'diff';
		case '.log':
			return 'plaintext';
		case '.txt':
			return 'plaintext';

		default:
			return 'plaintext';
	}
}
