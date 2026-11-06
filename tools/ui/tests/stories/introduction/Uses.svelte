<!--
	What this Storybook is for, as four tiles: designing in the Scrapbook, building from the components, checking with
	the accessibility addon and handing over to agents. Each has its icon, its name and one line.
-->
<script lang="ts">
	import { Blocks, CircleCheck, PenTool, Send } from '@lucide/svelte';

	const uses = [
		{
			icon: PenTool,
			line: 'Render variations in the Scrapbook, from the real system',
			name: 'Designing'
		},
		{
			icon: Blocks,
			line: 'Look up a component, its states and tokens, light and dark',
			name: 'Building'
		},
		{
			icon: CircleCheck,
			line: 'Every story is the real code, checked for accessibility',
			name: 'Checking'
		},
		{
			icon: Send,
			line: 'The rules export as the design system for Claude Design',
			name: 'Handing over'
		}
	];
</script>

<div class="uses">
	{#each uses as use (use.name)}
		<section class="use card-surface rounded-card">
			<span class="glyph rounded-control bg-surface-lifted elevated-control">
				<use.icon aria-hidden="true" size={18} />
			</span>

			<h3>{use.name}</h3>

			<p>{use.line}</p>
		</section>
	{/each}
</div>

<style>
	.uses {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr));
		gap: 8px;
	}

	.use {
		display: grid;
		gap: 8px;
		align-content: start;
		padding: 16px;
	}

	.glyph {
		display: grid;
		place-items: center;
		width: 36px;
		height: 36px;
		margin-bottom: 4px;
	}

	h3 {
		margin: 0;
		font-size: 16px;
		font-weight: 600;
	}

	p {
		margin: 0;
		color: var(--figure-muted);
		font-size: 13px;
		line-height: 1.45;
	}
</style>
