// Shared remark/rehype pipeline factory for MarkdownContent.
//
// The frozen plugin chain is expensive to build ( ~15 plugin instances ),
// and MarkdownContent used to rebuild it on every processMarkdown call:
// once per block at mount, and again on every coalesced chunk while
// streaming. Pipelines without attachments are shared process-wide per
// math flag; attachment-bearing pipelines are cached by the attachments
// array identity, which changes whenever extras are updated.

import { rehypeEnhanceCodeBlocks } from './plugins/rehype/enhance-code-blocks';
import { rehypeEnhanceLinks } from './plugins/rehype/enhance-links';
import { rehypeEnhanceMermaidBlocks } from './plugins/rehype/enhance-mermaid-blocks';
import { rehypeEnhanceSvgBlocks } from './plugins/rehype/enhance-svg-blocks';
import { rehypeFileBadge } from './plugins/rehype/file-badge';
import { rehypeMermaidPre } from './plugins/rehype/mermaid-pre';
import { rehypeRtlSupport } from './plugins/rehype/rehype-rtl-support';
import { rehypeResolveAttachmentImages } from './plugins/rehype/resolve-attachment-images';
import { rehypeSvgPre } from './plugins/rehype/svg-pre';
import { rehypeRestoreTableHtml } from './plugins/rehype/table-html-restorer';
import { remarkLiteralHtml } from './plugins/remark/literal-html';
import { FileTypeText } from '$lib/enums/files.enums';
import type { DatabaseMessageExtra } from '$lib/types/database';
import type { Root as HastRoot } from 'hast';
import { all as lowlightAll } from 'lowlight';
import type { Root as MdastRoot } from 'mdast';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';
import { remark } from 'remark';
import remarkBreaks from 'remark-breaks';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';

export interface MarkdownProcessor {
	parse(markdown: string): MdastRoot;
	run(tree: MdastRoot): Promise<HastRoot>;
	stringify(tree: HastRoot): string;
}

export interface MarkdownProcessorOptions {
	attachments?: DatabaseMessageExtra[];
	disableMath?: boolean;
	/** Render raw HTML found in the markdown instead of escaping it. */
	allowHtml?: boolean;
}

const sharedPipelines = new Map<string, MarkdownProcessor>();
const attachmentPipelines = new WeakMap<object, Map<string, MarkdownProcessor>>();

function buildPipeline({
	allowHtml = false,
	attachments,
	disableMath = false
}: MarkdownProcessorOptions): MarkdownProcessor {
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let proc: any = remark().use(remarkGfm); // GitHub Flavored Markdown

	if (!disableMath) {
		proc = proc.use(remarkMath); // Parse $inline$ and $$block$$ math
	}

	proc = proc.use(remarkBreaks); // Convert line breaks to <br>

	if (!allowHtml) {
		// Treat raw HTML as literal text with preserved indentation
		proc = proc.use(remarkLiteralHtml);
	}

	// Convert Markdown AST to rehype. Keep raw HTML as-is when allowHtml is set.
	proc = proc.use(remarkRehype, allowHtml ? { allowDangerousHtml: true } : undefined);

	if (!disableMath) {
		proc = proc.use(rehypeKatex); // Render math using KaTeX
	}

	const pipeline = proc
		.use(rehypeHighlight, {
			aliases: { [FileTypeText.XML]: [FileTypeText.SVELTE, FileTypeText.VUE] },
			languages: lowlightAll
		}) // Add syntax highlighting
		.use(rehypeRestoreTableHtml) // Restore limited HTML (e.g. <br>, <ul>) inside Markdown tables
		.use(rehypeEnhanceLinks) // Add target="_blank" to links
		.use(rehypeFileBadge) // Render file:// anchors as inline badge chips
		.use(rehypeMermaidPre) // Convert mermaid blocks to <pre class="mermaid">
		.use(rehypeSvgPre) // Convert svg blocks to <pre class="svg-block">
		.use(rehypeEnhanceCodeBlocks) // Wrap code blocks with header and actions
		.use(rehypeEnhanceMermaidBlocks) // Wrap mermaid blocks with header and actions
		.use(rehypeEnhanceSvgBlocks) // Wrap svg blocks with header and actions
		.use(rehypeResolveAttachmentImages, { attachments })
		.use(rehypeRtlSupport) // Add bidirectional text support
		.use(rehypeStringify, { allowDangerousHtml: true }); // Convert to HTML string

	return pipeline as MarkdownProcessor;
}

export function getMarkdownProcessor(options: MarkdownProcessorOptions): MarkdownProcessor {
	if (options.attachments && options.attachments.length > 0) {
		let byOptions = attachmentPipelines.get(options.attachments);

		if (!byOptions) {
			byOptions = new Map<string, MarkdownProcessor>();
			attachmentPipelines.set(options.attachments, byOptions);
		}

		const key = `${Boolean(options.disableMath)}:${Boolean(options.allowHtml)}`;

		let cached = byOptions.get(key);

		if (!cached) {
			cached = buildPipeline(options);
			byOptions.set(key, cached);
		}

		return cached;
	}

	const key = `${Boolean(options.disableMath)}:${Boolean(options.allowHtml)}`;

	let cached = sharedPipelines.get(key);

	if (!cached) {
		cached = buildPipeline(options);
		sharedPipelines.set(key, cached);
	}

	return cached;
}
