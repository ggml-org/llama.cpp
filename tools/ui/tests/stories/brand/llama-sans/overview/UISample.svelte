<!-- Llama Sans UI at its working sizes: a row of controls at the font's own line height, then a paragraph at 1.5. -->
<div class="ui-sample">
	<div class="controls">
		<span class="button primary"> New project</span>

		<span class="button">Cancel</span>

		<span class="chip">v0.6 · 975 glyphs</span>
	</div>

	<p>
		The labels on a button and the paragraphs of an article can share a font. UI is spaced and
		kerned like the system fonts at their text sizes, and its line box is centred on the cap height,
		so capitals and icons sit in the middle of every control.
	</p>
</div>

<style>
	.ui-sample {
		display: grid;
		gap: 24px;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		align-items: center;
		font: 500 14px / normal var(--ui);
	}
	.button {
		padding: 8px 14px;
		border: 1px solid var(--rule);
		border-radius: 8px;
	}
	.primary {
		border-color: var(--ink);
		background: var(--ink);
		color: var(--paper);
	}
	.chip {
		padding: 3px 9px;
		border-radius: 999px;
		background: var(--sheet);
		font-size: 12px;
		color: var(--muted);
	}
	p {
		margin: 0;
		font: 400 16px / 1.5 var(--ui);
	}
</style>
