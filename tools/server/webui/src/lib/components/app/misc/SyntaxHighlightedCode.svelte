<script lang="ts">
	import hljs from 'highlight.js';
	import { browser } from '$app/environment';
	import { mode } from 'mode-watcher';

	import githubDarkCss from 'highlight.js/styles/github-dark.css?inline';
	import githubLightCss from 'highlight.js/styles/github.css?inline';

	interface Props {
		code: string;
		language?: string;
		class?: string;
		maxHeight?: string;
		maxWidth?: string;
	}

	let {
		code,
		language = 'text',
		class: className = '',
		maxHeight = '60vh',
		maxWidth = ''
	}: Props = $props();

	let highlightedHtml = $state('');

	function loadHighlightTheme(isDark: boolean) {
		if (!browser) return;

		const existingThemes = document.querySelectorAll('style[data-highlight-theme-preview]');
		existingThemes.forEach((style) => style.remove());

		const style = document.createElement('style');
		style.setAttribute('data-highlight-theme-preview', 'true');
		style.textContent = isDark ? githubDarkCss : githubLightCss;

		document.head.appendChild(style);
	}

	$effect(() => {
		const currentMode = mode.current;
		const isDark = currentMode === 'dark';

		loadHighlightTheme(isDark);
	});

	$effect(() => {
		if (!code) {
			highlightedHtml = '';
			return;
		}

		try {
			// Check if the language is supported
			const lang = language.toLowerCase();
			const isSupported = hljs.getLanguage(lang);

			if (isSupported) {
				const result = hljs.highlight(code, { language: lang });
				highlightedHtml = result.value;
			} else {
				// Try auto-detection or fallback to plain text
				const result = hljs.highlightAuto(code);
				highlightedHtml = result.value;
			}
		} catch {
			// Fallback to escaped plain text
			highlightedHtml = code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
		}
	});
</script>

<div
	class="code-preview-wrapper rounded-lg border border-border bg-muted {className}"
	style="max-height: {maxHeight}; max-width: {maxWidth};"
>
	<!-- Needs to be formatted as single line for proper rendering -->
	<pre class="m-0"><code class="hljs text-sm leading-relaxed">{@html highlightedHtml}</code></pre>
</div>

<style>
	.code-preview-wrapper {
		font-family:
			ui-monospace, SFMono-Regular, 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas,
			'Liberation Mono', Menlo, monospace;
	}

	.code-preview-wrapper pre {
		background: transparent;
	}

	.code-preview-wrapper code {
		background: transparent;
	}
</style>
