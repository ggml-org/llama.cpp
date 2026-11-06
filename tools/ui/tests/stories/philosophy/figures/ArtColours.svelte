<!--
	The communications palette beyond the greys, as grounds: sand (silicon's raw material), the day sky and the night
	sky, and cinder, the one warm accent. Each swatch is the real token, with its name and what it stands for.
-->
<script lang="ts">
	const grounds = [
		{ dark: false, name: 'Sand', note: 'Silicon, the base of every chip', token: '--sand-100' },
		{ dark: false, name: 'Day sky', note: 'llama.app’s light blue', token: '--art-day-sky' },
		{ dark: true, name: 'Night sky', note: 'Its navy, after dark', token: '--art-night-sky' },
		{ dark: true, name: 'Cinder', note: 'The one accent, #ee6c1e', token: '--art-cinder' }
	];
</script>

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

<style>
	.grounds {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr));
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
</style>
