<!-- Llama Sans UI in its element: controls and small text at the font's own line height (1.25 em, cap-centred). -->
<script lang="ts">
	import InterfaceSample from './InterfaceSample.svelte';

	let { glyphCount, guides }: { glyphCount: number; guides: boolean } = $props();

	const tabs = ['Overview', 'Glyphs', 'Kerning', 'Export'];
	let tab = $state('Overview');
	let name = $state('Llama Sans');
	const settings = $state([
		{ hint: 'Align numbers in tables', label: 'Tabular figures', on: true },
		{ hint: 'Raise punctuation between capitals', label: 'Case-sensitive forms', on: false },
		{ hint: 'Off: outlines changed from the source', label: 'Hinting', on: false }
	]);
	const rows = $derived([
		{ glyphs: glyphCount, size: '102.3 KB', style: 'Regular', weight: 400 },
		{ glyphs: glyphCount, size: '102.5 KB', style: 'Medium', weight: 500 },
		{ glyphs: glyphCount, size: '102.3 KB', style: 'Bold', weight: 700 }
	]);
</script>

<div class:guides class="ui-showcase">
	<InterfaceSample {glyphCount} {guides} />

	<div class="panels">
		<form class="panel" onsubmit={(e) => e.preventDefault()}>
			<div aria-label="Sections" class="tabs" role="tablist">
				{#each tabs as t (t)}
					<button
						class:active={tab === t}
						aria-selected={tab === t}
						onclick={() => (tab = t)}
						role="tab"
						type="button"
					>
						<span class="t">{t}</span>
					</button>
				{/each}
			</div>

			<label class="field" for="family-name">
				<span class="field-label t">Family name</span>

				<input bind:value={name} class="input" id="family-name" />

				<span class="help t">Shown in font menus. {name.length} / 31 characters.</span>
			</label>

			<div class="field">
				<span class="field-label t">Formats</span>

				<div class="seg">
					<span class="seg-item on t">TTF</span><span class="seg-item t">WOFF2</span><span
						class="seg-item t">Variable</span
					>
				</div>
			</div>

			<div class="form-actions">
				<button class="btn ghost" type="button"><span class="t">Reset</span></button>

				<button class="btn solid" type="submit"><span class="t">Save changes</span></button>
			</div>
		</form>

		<div class="panel">
			<p class="panel-title t">Font features</p>

			{#each settings as s (s.label)}
				<label class="setting">
					<span class="setting-text">
						<span class="setting-label t">{s.label}</span>

						<span class="help t">{s.hint}</span>
					</span>

					<input bind:checked={s.on} aria-label={s.label} class="switch" type="checkbox" />
				</label>
			{/each}

			<div class="toast" role="status">
				<span aria-hidden="true" class="dot"></span>

				<span class="toast-text t">Build finished · 3 weights · 1.4 s</span>

				<button class="link" type="button"><span class="t">Undo</span></button>
			</div>
		</div>

		<div class="panel wide">
			<p class="panel-title t">Static fonts</p>

			<table>
				<thead>
					<tr
						><th><span class="t">Style</span></th><th class="num"><span class="t">Weight</span></th
						><th class="num"><span class="t">Glyphs</span></th><th class="num"
							><span class="t">Size</span></th
						></tr
					>
				</thead>

				<tbody>
					{#each rows as r (r.style)}
						<tr>
							<td><span class="t">{r.style}</span></td>

							<td class="num"><span class="t">{r.weight}</span></td>

							<td class="num"><span class="t">{r.glyphs}</span></td>

							<td class="num"><span class="t">{r.size}</span></td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</div>
</div>

<style>
	.ui-showcase {
		display: grid;
		gap: 24px;
		font-family: var(--display);
	}
	.panels {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
		gap: 16px;
	}
	.panel {
		display: grid;
		gap: 14px;
		align-content: start;
		min-width: 0;
		padding: 16px;
		border: 1px solid var(--rule);
		border-radius: 10px;
		background: var(--sheet);
		font: 400 14px / normal var(--display);
	}
	.panel.wide {
		grid-column: 1 / -1;
	}
	.panel-title {
		margin: 0;
		font: 600 15px / normal var(--display);
	}
	.tabs {
		display: flex;
		gap: 2px;
		padding: 3px;
		border-radius: 8px;
		background: var(--paper);
		overflow-x: auto;
	}
	.tabs button {
		flex: 1;
		padding: 0.45em 0.8em;
		border: 0;
		border-radius: 8px;
		background: none;
		color: var(--muted);
		font: 500 13px / normal var(--display);
		cursor: pointer;
	}
	.tabs button.active {
		background: var(--sheet);
		color: var(--ink);
		box-shadow: 0 0 0 1px var(--rule);
	}
	.field {
		display: grid;
		gap: 6px;
	}
	.field-label {
		font: 500 13px / normal var(--display);
	}
	.input {
		padding: 0.55em 0.75em;
		border: 1px solid var(--rule);
		border-radius: 8px;
		background: var(--paper);
		color: var(--ink);
		font: 400 14px / normal var(--display);
	}
	.input:focus-visible {
		outline: 2px solid var(--guide);
		outline-offset: 1px;
	}
	.help {
		font: 400 12px / normal var(--display);
		color: var(--muted);
	}
	.seg {
		display: inline-flex;
		justify-self: start;
		border: 1px solid var(--rule);
		border-radius: 8px;
		overflow: hidden;
	}
	.seg-item {
		padding: 0.35em 0.8em;
		font: 500 12px / normal var(--display);
		color: var(--muted);
	}
	.seg-item + .seg-item {
		border-left: 1px solid var(--rule);
	}
	.seg-item.on {
		background: var(--ink);
		color: var(--paper);
	}
	.form-actions {
		display: flex;
		gap: 8px;
		justify-content: flex-end;
	}
	.btn {
		padding: 0.5em 0.95em;
		border-radius: 8px;
		font: 500 13px / normal var(--display);
		cursor: pointer;
	}
	.btn.ghost {
		border: 1px solid var(--rule);
		background: none;
		color: var(--ink);
	}
	.btn.solid {
		border: 1px solid var(--ink);
		background: var(--ink);
		color: var(--paper);
	}
	.setting {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 12px;
		padding-block: 6px;
		border-top: 1px solid var(--rule);
		cursor: pointer;
	}
	.setting-text {
		display: grid;
		gap: 2px;
	}
	.setting-label {
		font: 500 14px / normal var(--display);
	}
	.switch {
		appearance: none;
		flex: none;
		width: 34px;
		height: 20px;
		margin: 0;
		border-radius: 999px;
		background: var(--rule);
		position: relative;
		cursor: pointer;
		transition: background 0.15s;
	}
	.switch::after {
		content: '';
		position: absolute;
		top: 2px;
		left: 2px;
		width: 16px;
		height: 16px;
		border-radius: 50%;
		background: var(--sheet);
		transition: transform 0.15s;
	}
	.switch:checked {
		background: var(--guide);
	}
	.switch:checked::after {
		transform: translateX(14px);
	}
	.switch:focus-visible {
		outline: 2px solid var(--guide);
		outline-offset: 2px;
	}
	.toast {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 10px 12px;
		border-radius: 8px;
		background: var(--ink);
		color: var(--paper);
		font: 500 13px / normal var(--display);
	}
	.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--guide);
		flex: none;
	}
	.toast-text {
		flex: 1;
	}
	.link {
		border: 0;
		background: none;
		color: inherit;
		font: inherit;
		text-decoration: underline;
		cursor: pointer;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font: 400 13px / normal var(--display);
		font-variant-numeric: tabular-nums;
	}
	th {
		text-align: left;
		font-weight: 500;
		color: var(--muted);
	}
	th,
	td {
		padding: 8px 10px 8px 0;
		border-bottom: 1px solid var(--rule);
	}
	.num {
		text-align: right;
	}
	.guides .panels .t {
		outline: 1px dashed var(--guide);
	}
</style>
