<!-- The family's numbers: vertical metrics, the weight axis, stems, and each version's spacing. -->
<script lang="ts">
	import { specimen } from '../specimen';
	const { axis, metrics, styles } = specimen;
</script>

<dl class="metrics">
	<div>
		<dt>UPM</dt>

		<dd>{metrics.upm}</dd>
	</div>

	<div>
		<dt>Cap height</dt>

		<dd>{metrics.cap}</dd>
	</div>

	<div>
		<dt>x-height</dt>

		<dd>{metrics.x}</dd>
	</div>

	<div>
		<dt>Ascender</dt>

		<dd>{metrics.asc}</dd>
	</div>

	<div>
		<dt>Descender</dt>

		<dd>{metrics.dsc}</dd>
	</div>

	<div>
		<dt>Weight axis</dt>

		<dd>{axis.min}–{axis.max}</dd>
	</div>

	<div>
		<dt>Stems {styles.map((s) => s.weight).join(' · ')}</dt>

		<dd>{styles.map((s) => s.stem).join(' · ')}</dd>
	</div>

	<div>
		<dt>Display</dt>

		<dd>tracking −26 · kerning 115% · line 1.15</dd>
	</div>

	<div>
		<dt>UI</dt>

		<dd>tracking +3 · kerning after SF Pro · line 1.25, cap-centred</dd>
	</div>
</dl>

<style>
	.metrics {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 28px;
		margin: 0;
		padding: 14px 0 0;
		border-top: 1px solid var(--rule);
	}
	.metrics div {
		display: grid;
		gap: 2px;
	}
	dt {
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--muted);
	}
	dd {
		margin: 0;
		font: 400 15px var(--mono);
		font-variant-numeric: tabular-nums;
	}
	/* on phones the page is for looking at and trying the type, so the metrics step aside */
	@media (max-width: 600px) {
		.metrics {
			display: none;
		}
	}
</style>
