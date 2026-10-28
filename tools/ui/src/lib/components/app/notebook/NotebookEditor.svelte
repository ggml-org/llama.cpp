<script lang="ts">
	import { Prec } from '@codemirror/state';
	import { highlightActiveLine, keymap } from '@codemirror/view';
	import { MarkdownEditor } from '$lib/components/app';

	interface Props {
		class?: string;
		onChange?: (value: string) => void;
		placeholder?: string;
		value: string;
	}

	let { class: className = '', onChange, placeholder = '', value }: Props = $props();

	let editor: MarkdownEditor | undefined = $state();

	const extensions = [
		// Shift+Enter belongs to the notebook (generate), not to a new line
		Prec.highest(keymap.of([{ key: 'Shift-Enter', run: () => true }])),
		highlightActiveLine()
	];

	/** Focus the editor with the cursor at the end of the text. */
	export function focusEnd() {
		editor?.focusAt();
	}
</script>

<div class="notebook-editor {className}">
	<MarkdownEditor bind:this={editor} {extensions} {onChange} {placeholder} {value} />
</div>

<style>
	.notebook-editor {
		line-height: 1.75;
	}

	/* The editor grows with its text; the notebook page is the scroller */
	.notebook-editor :global(.cm-scroller) {
		overflow: visible;
	}

	/* Same text column as chat messages */
	.notebook-editor :global(.cm-content) {
		width: 100%;
		max-width: 48rem;
		margin: 0 auto;
	}

	/* Empty notebook: just the caret and the placeholder, like a textarea */
	.notebook-editor :global(.cm-activeLine:has(.cm-placeholder)) {
		background-color: transparent;
	}

	/* Line under the cursor: barely tinted, like our inputs */
	.notebook-editor :global(.cm-activeLine) {
		background-color: color-mix(in oklch, var(--muted) 50%, transparent);
	}

	.notebook-editor :global(.cm-md-raw-code.cm-activeLine) {
		background: color-mix(in oklch, var(--muted) 50%, var(--code-background));
	}
</style>
