<!--
	Proposal D · Plain words, exact settings: a short list of what people want, each in their words with a real control
	(Switch, ToggleGroup), and the exact setting it stands for in a second column. The same choice, two vocabularies.
-->
<script lang="ts">
	import { Switch } from '$lib/components/ui/switch';
	import * as ToggleGroup from '$lib/components/ui/toggle-group';
</script>

<div class="plain card-surface rounded-card" inert>
	<div class="row head">
		<p class="first">What most people see</p>

		<p>What a power user can set</p>
	</div>

	<div class="row">
		<div class="say">
			<span>Look at my photos</span>

			<Switch checked />
		</div>

		<span class="exact">--mmproj mmproj-F16.gguf</span>
	</div>

	<div class="row">
		<div class="say">
			<span>Remember long letters</span>

			<Switch checked />
		</div>

		<span class="exact">--ctx-size 32768</span>
	</div>

	<div class="row">
		<div class="say">
			<span>Answers</span>

			<ToggleGroup.Root size="sm" type="single" value="balanced">
				<ToggleGroup.Item value="exact">Exact</ToggleGroup.Item>

				<ToggleGroup.Item value="balanced">Balanced</ToggleGroup.Item>

				<ToggleGroup.Item value="creative">Creative</ToggleGroup.Item>
			</ToggleGroup.Root>
		</div>

		<span class="exact">--temp 0.2 · 0.7 · 1.1</span>
	</div>

	<div class="row">
		<div class="say">
			<span>Fits this laptop</span>

			<span class="ok">8 GB free</span>
		</div>

		<span class="exact">model.Q4_K_M.gguf · 4.6 GB</span>
	</div>
</div>

<style>
	.plain {
		display: grid;
		padding: 8px 16px;
	}

	.row {
		display: grid;
		grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
		align-items: center;
		gap: 16px;
		padding: 12px 0;
		border-top: 1px solid var(--line-default);
	}

	.row.head {
		border-top: 0;
	}

	p {
		margin: 0;
		color: var(--figure-muted);
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	p.first {
		color: var(--art-cinder);
	}

	.say {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		font-size: 14px;
	}

	.ok {
		color: var(--figure-muted);
		font-size: 13px;
	}

	.exact {
		overflow: hidden;
		color: var(--figure-muted);
		font: 500 12px var(--mono);
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	@media (width < 640px) {
		.row {
			grid-template-columns: minmax(0, 1fr);
			gap: 6px;
		}

		.row.head p:last-child {
			display: none;
		}
	}
</style>
