<!--
	The art colours as grounds and marks: sand (silicon's raw material), sky and navy (the day and the night sky), then
	cinder, the one warm accent, and blue, green and amber for shapes and numerals. Each swatch is the real token, with
	its name and what it stands for.
-->
<script lang="ts">
	const grounds = [
		{ dark: false, name: 'Sand', note: 'Silicon, the base of every chip', token: '--sand-100' },
		{ dark: false, name: 'Sky', note: 'The day sky', token: '--art-sky' },
		{ dark: true, name: 'Navy', note: 'The night sky', token: '--art-navy' }
	];

	const marks = [
		{ name: 'Cinder', token: '--art-cinder' },
		{ name: 'Blue', token: '--art-blue' },
		{ name: 'Green', token: '--art-green' },
		{ name: 'Amber', token: '--art-amber' }
	];
</script>

<div class="art">
	<div class="grounds">
		{#each grounds as ground (ground.name)}
			<div
				style:background="var({ground.token})"
				class={['ground rounded-card', { dark: ground.dark }]}
			>
				<b>{ground.name}</b>

				<span>{ground.note}</span>

				<span class="token">{ground.token}</span>
			</div>
		{/each}
	</div>

	<div class="marks">
		{#each marks as mark (mark.name)}
			<div class="mark">
				<i style:background="var({mark.token})" class="dot"></i>

				<span>{mark.name}</span>
			</div>
		{/each}
	</div>
</div>

<style>
	.art {
		display: grid;
		gap: 16px;
	}

	.grounds {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr));
		gap: 8px;
	}

	.ground {
		display: grid;
		align-content: end;
		gap: 4px;
		min-height: 140px;
		padding: 16px;
		color: #111;
	}

	.ground.dark {
		color: #fafafa;
	}

	b {
		font-size: 20px;
		font-weight: 600;
	}

	.ground span {
		font-size: 13px;
		opacity: 0.8;
	}

	.ground .token {
		font: 500 11px var(--mono);
		opacity: 0.7;
	}

	.marks {
		display: flex;
		flex-wrap: wrap;
		gap: 20px;
	}

	.mark {
		display: flex;
		align-items: center;
		gap: 8px;
		color: var(--figure-muted);
		font-size: 13px;
	}

	.dot {
		width: 20px;
		height: 20px;
		border-radius: 50%;
	}
</style>
