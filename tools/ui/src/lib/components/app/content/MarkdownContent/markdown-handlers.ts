/**
 * Event handler factories for markdown content components.
 * Uses dependency injection pattern to avoid direct component state access.
 */

import { copyCodeToClipboard, copyToClipboard } from '$lib/utils';
import { MERMAID_WRAPPER_CLASS, MERMAID_BLOCK_CLASS, MERMAID_SYNTAX_ATTR } from '$lib/constants';

export interface PreviewState {
	previewDialogOpen: boolean;
	previewCode: string;
	previewLanguage: string;
	setPreviewDialogOpen: (open: boolean) => void;
	setPreviewCode: (code: string) => void;
	setPreviewLanguage: (lang: string) => void;
}

export interface MermaidPreviewState {
	mermaidPreviewOpen: boolean;
	mermaidPreviewSvgHtml: string;
	setMermaidPreviewOpen: (open: boolean) => void;
	setMermaidPreviewSvgHtml: (html: string) => void;
}

export interface RenderedBlocksState {
	renderedBlocks: Array<{ id: string; html: string; contentHash?: string }>;
	setRenderedBlocks: (blocks: Array<{ id: string; html: string; contentHash?: string }>) => void;
}

/**
 * Creates a click handler for copy buttons in code blocks.
 * Copies the code content to clipboard.
 */
export function createHandleCopyClick() {
	return async function handleCopyClick(event: Event) {
		event.preventDefault();
		event.stopPropagation();

		const target = event.currentTarget as HTMLButtonElement | null;
		if (!target) return;

		const wrapper = target.closest('.code-block-wrapper');
		if (!wrapper) return;

		const codeElement = wrapper.querySelector<HTMLElement>('code[data-code-id]');
		if (!codeElement) return;

		const rawCode = codeElement.textContent ?? '';

		try {
			await copyCodeToClipboard(rawCode);
		} catch (error) {
			console.error('Failed to copy code:', error);
		}
	};
}

/**
 * Creates a handler for preview dialog open state changes.
 * Clears preview content when dialog is closed.
 */
export function createHandlePreviewDialogOpenChange(previewState: PreviewState) {
	return function handlePreviewDialogOpenChange(open: boolean) {
		previewState.setPreviewDialogOpen(open);

		if (!open) {
			previewState.setPreviewCode('');
			previewState.setPreviewLanguage('text');
		}
	};
}

/**
 * Creates a click handler for preview buttons within HTML code blocks.
 * Opens a preview dialog with the rendered HTML content.
 */
export function createHandlePreviewClick(previewState: PreviewState) {
	return async function handlePreviewClick(event: Event) {
		event.preventDefault();
		event.stopPropagation();

		const target = event.currentTarget as HTMLButtonElement | null;
		if (!target) return;

		const wrapper = target.closest('.code-block-wrapper');
		if (!wrapper) return;

		const codeElement = wrapper.querySelector<HTMLElement>('code[data-code-id]');
		if (!codeElement) return;

		const rawCode = codeElement.textContent ?? '';
		const languageLabel = wrapper.querySelector<HTMLElement>('.code-language');
		const language = languageLabel?.textContent?.trim() || 'text';

		previewState.setPreviewCode(rawCode);
		previewState.setPreviewLanguage(language);
		previewState.setPreviewDialogOpen(true);
	};
}

/**
 * Creates a click handler for mermaid block interactions.
 * Handles copy, preview, and diagram click events via event delegation.
 */
export function createHandleMermaidClick(mermaidState: MermaidPreviewState) {
	return async function handleMermaidClick(event: MouseEvent) {
		const target = event.target as HTMLElement;

		// Check if clicking on copy or preview button in mermaid block
		const copyBtn = target.closest(`.${MERMAID_WRAPPER_CLASS} .copy-code-btn`);
		const previewBtn = target.closest(`.${MERMAID_WRAPPER_CLASS} .preview-code-btn`);

		if (copyBtn || previewBtn) {
			const wrapper = target.closest(`.${MERMAID_WRAPPER_CLASS}`);
			if (!wrapper) return;

			const preElement = wrapper.querySelector<HTMLElement>(
				`pre.${MERMAID_BLOCK_CLASS}[${MERMAID_SYNTAX_ATTR}]`
			);
			if (!preElement) return;

			const mermaidSyntax = preElement.getAttribute(MERMAID_SYNTAX_ATTR) ?? '';

			if (copyBtn) {
				event.preventDefault();
				event.stopPropagation();
				try {
					await copyToClipboard(mermaidSyntax);
				} catch (error) {
					console.error('Failed to copy mermaid syntax:', error);
				}
				return;
			}

			if (previewBtn) {
				event.preventDefault();
				event.stopPropagation();
				const svg = preElement.querySelector('svg');
				if (!svg) return;
				mermaidState.setMermaidPreviewSvgHtml(svg.outerHTML);
				mermaidState.setMermaidPreviewOpen(true);
				return;
			}
		}

		// Otherwise, open preview when clicking on the mermaid diagram itself
		const mermaidEl = target.closest(`.${MERMAID_BLOCK_CLASS}`);
		if (!mermaidEl) return;

		const svg = mermaidEl.querySelector('svg');
		if (!svg) return;

		mermaidState.setMermaidPreviewSvgHtml(svg.outerHTML);
		mermaidState.setMermaidPreviewOpen(true);
	};
}

/**
 * Creates a handler for mermaid preview dialog open state changes.
 * Cleans up SVG content when dialog is closed.
 */
export function createHandleMermaidPreviewOpenChange(mermaidState: MermaidPreviewState) {
	return function handleMermaidPreviewOpenChange(open: boolean) {
		mermaidState.setMermaidPreviewOpen(open);
		if (!open) {
			mermaidState.setMermaidPreviewSvgHtml('');
		}
	};
}

/**
 * Creates an error handler for images that fail to load (e.g., CORS issues).
 * Shows fallback UI for broken images.
 */
export function createHandleImageError(
	renderedBlocksState: RenderedBlocksState,
	IMAGE_NOT_ERROR_BOUND_SELECTOR: string,
	DATA_ERROR_BOUND_ATTR: string,
	BOOL_TRUE_STRING: string
) {
	return async function handleImageError(event: Event) {
		const img = event.target as HTMLImageElement;
		if (!img) return;

		const blockId = img.closest('[data-block-id]')?.getAttribute('data-block-id');
		if (!blockId) return;

		const block = renderedBlocksState.renderedBlocks.find((b) => b.id === blockId);
		if (!block) return;

		// Skip if already handled
		if (img.dataset[DATA_ERROR_BOUND_ATTR] === BOOL_TRUE_STRING) return;
		img.dataset[DATA_ERROR_BOUND_ATTR] = BOOL_TRUE_STRING;

		// Get the fallback HTML and replace the image
		const fallbackHtml = `<div class="image-error-placeholder" data-original-src="${img.src}">
			<span class="image-error-icon">⚠️</span>
			<span class="image-error-text">Failed to load image</span>
		</div>`;

		// Replace the img element with fallback in the block's HTML
		const newHtml = block.html.replace(/img[^>]*src=["']([^"']*)[^>]*>/g, (match, src) => {
			if (src === img.src) {
				return fallbackHtml.replace('data-original-src=""', `data-original-src="${src}"`);
			}
			return match;
		});

		// Update the block
		const newBlocks = renderedBlocksState.renderedBlocks.map((b) =>
			b.id === blockId ? { ...b, html: newHtml } : b
		);
		renderedBlocksState.setRenderedBlocks(newBlocks);
	};
}

/**
 * Creates a function to set up code block action event listeners.
 * Binds click handlers to copy and preview buttons within code blocks.
 */
export function createSetupCodeBlockActions(
	handleCopyClick: (event: Event) => void,
	handlePreviewClick: (event: Event) => void
) {
	return function setupCodeBlockActions(containerRef: HTMLElement | null) {
		if (!containerRef) return;

		const wrappers = containerRef.querySelectorAll<HTMLElement>('.code-block-wrapper');

		for (const wrapper of wrappers) {
			const copyButton = wrapper.querySelector<HTMLButtonElement>('.copy-code-btn');
			const previewButton = wrapper.querySelector<HTMLButtonElement>('.preview-code-btn');

			if (copyButton && copyButton.dataset.listenerBound !== 'true') {
				copyButton.dataset.listenerBound = 'true';
				copyButton.addEventListener('click', handleCopyClick);
			}

			if (previewButton && previewButton.dataset.listenerBound !== 'true') {
				previewButton.dataset.listenerBound = 'true';
				previewButton.addEventListener('click', handlePreviewClick);
			}
		}
	};
}

/**
 * Creates a function to set up image error handlers.
 * Attaches error handlers to images to show fallback UI when loading fails.
 */
export function createSetupImageErrorHandlers(
	handleImageError: (event: Event) => void,
	IMAGE_NOT_ERROR_BOUND_SELECTOR: string,
	DATA_ERROR_BOUND_ATTR: string,
	BOOL_TRUE_STRING: string
) {
	return function setupImageErrorHandlers(containerRef: HTMLElement | null) {
		if (!containerRef) return;

		const images = containerRef.querySelectorAll<HTMLImageElement>(IMAGE_NOT_ERROR_BOUND_SELECTOR);

		for (const img of images) {
			img.dataset[DATA_ERROR_BOUND_ATTR] = BOOL_TRUE_STRING;
			img.addEventListener('error', handleImageError);
		}
	};
}
