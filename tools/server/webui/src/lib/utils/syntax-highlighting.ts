import {
	createHighlighter,
	type Highlighter,
	type BundledLanguage,
	type BundledTheme
} from 'shiki';

let highlighter: Highlighter | null = null;

// Initialize the highlighter with common languages and themes
export async function initHighlighter(): Promise<Highlighter> {
	if (highlighter) {
		return highlighter;
	}

	try {
		highlighter = await createHighlighter({
			themes: ['github-dark', 'github-light'],
			langs: [
				'javascript',
				'typescript',
				'python',
				'java',
				'cpp',
				'c',
				'rust',
				'go',
				'html',
				'css',
				'json',
				'markdown',
				'bash',
				'shell',
				'sql',
				'yaml',
				'xml',
				'php',
				'ruby',
				'swift',
				'kotlin',
				'dart',
				'r',
				'scala',
				'clojure',
				'haskell',
				'lua',
				'perl',
				'powershell',
				'dockerfile',
				'nginx',
				'apache',
				'diff'
			] as BundledLanguage[]
		});
		return highlighter!;
	} catch (error) {
		console.error('Failed to initialize syntax highlighter:', error);
		throw error;
	}
}

// Normalize code indentation by removing common leading whitespace
function normalizeCodeIndentation(code: string): string {
	const lines = code.split('\n');

	// Skip empty lines when calculating minimum indentation
	const nonEmptyLines = lines.filter((line) => line.trim().length > 0);

	if (nonEmptyLines.length === 0) {
		return code;
	}

	// Find the minimum indentation (number of leading spaces/tabs)
	const minIndent = Math.min(
		...nonEmptyLines.map((line) => {
			const match = line.match(/^(\s*)/);
			return match ? match[1].length : 0;
		})
	);

	// Remove the common leading whitespace from all lines
	if (minIndent > 0) {
		return lines
			.map((line) => (line.length > minIndent ? line.slice(minIndent) : line))
			.join('\n');
	}

	return code;
}

// Highlight code with language detection
export async function highlightCode(
	code: string,
	language?: string,
	theme: BundledTheme = 'github-dark'
): Promise<string> {
	try {
		const hl = await initHighlighter();

		// Normalize indentation to fix first-line indentation issues
		const normalizedCode = normalizeCodeIndentation(code);

		// If no language specified, try to detect common patterns
		const detectedLang = language || detectLanguage(normalizedCode);

		// Check if the language is supported
		const supportedLangs = hl.getLoadedLanguages();
		const langToUse = supportedLangs.includes(detectedLang as BundledLanguage)
			? (detectedLang as BundledLanguage)
			: 'text';

		const highlighted = hl.codeToHtml(normalizedCode, {
			lang: langToUse,
			theme: theme
		});

		// Wrap the highlighted code with language label and copy button
		return wrapCodeWithControls(highlighted, langToUse, normalizedCode);
	} catch (error) {
		console.error('Syntax highlighting failed:', error);
		// Fallback to plain code block with controls
		const fallbackCode = `<pre><code>${escapeHtml(code)}</code></pre>`;
		return wrapCodeWithControls(fallbackCode, 'text', code);
	}
}

// Simple language detection based on code patterns
function detectLanguage(code: string): string {
	const trimmed = code.trim();

	// JavaScript/TypeScript patterns
	if (
		/^(import|export|const|let|var|function|class|interface|type)\s/.test(trimmed) ||
		/console\.log|document\.|window\.|require\(/.test(code) ||
		/\.(js|ts|jsx|tsx)$/.test(trimmed)
	) {
		return /interface|type\s+\w+\s*=|:\s*\w+/.test(code) ? 'typescript' : 'javascript';
	}

	// Python patterns
	if (
		/^(def|class|import|from|if __name__|print\(|lambda)/.test(trimmed) ||
		/:\s*$/.test(trimmed.split('\n')[0]) ||
		/\.py$/.test(trimmed)
	) {
		return 'python';
	}

	// HTML patterns
	if (/<\/?[a-z][\s\S]*>/i.test(code) || /<!DOCTYPE|<html|<head|<body/.test(code)) {
		return 'html';
	}

	// CSS patterns
	if (/\{[^}]*\}/.test(code) && /[.#]?[\w-]+\s*\{/.test(code)) {
		return 'css';
	}

	// JSON patterns
	if (/^\s*[{[]/.test(trimmed) && /[}]]\s*$/.test(trimmed)) {
		try {
			JSON.parse(code);
			return 'json';
		} catch {
			// Not valid JSON
		}
	}

	// Shell/Bash patterns
	if (
		/^#!\/bin\//.test(trimmed) ||
		/^#!\s*\/bin\/(bash|sh)/.test(trimmed) ||
		/^\s*(cd|ls|mkdir|rm|cp|mv|grep|awk|sed|curl|wget)\s/.test(trimmed) ||
		/\${|\$\(|\$\w+/.test(code)
	) {
		return 'bash';
	}

	// SQL patterns
	if (/^(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s/i.test(trimmed)) {
		return 'sql';
	}

	// Default to text
	return 'text';
}

// Escape HTML entities (works in both browser and server)
function escapeHtml(text: string): string {
	return text
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;')
		.replace(/'/g, '&#39;');
}

// Wrap highlighted code with language label and copy button
function wrapCodeWithControls(highlightedHtml: string, language: string, rawCode: string): string {
	// Generate a unique ID for this code block
	const codeId = `code-${Math.random().toString(36).substr(2, 9)}`;

	// Create the wrapper with language label and copy button
	return `
		<div class="code-block-wrapper relative">
			<div class="code-block-header absolute top-0 left-0 right-0 z-10 flex justify-between items-center px-4 py-2">
				<span class="text-2xs text-muted-foreground font-mono uppercase">${language}</span>
				<button 
					class="copy-code-btn text-muted-foreground hover:text-foreground transition-colors p-1 rounded" 
					data-code-id="${codeId}"
					title="Copy to clipboard"
				>
					<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
					</svg>
				</button>
			</div>
			<div class="code-content" data-code-id="${codeId}" data-raw-code="${escapeHtml(rawCode)}">
				${highlightedHtml}
			</div>
		</div>
	`;
}

// Get theme based on current mode
export function getThemeForMode(isDark: boolean): BundledTheme {
	return isDark ? 'github-dark' : 'github-light';
}
