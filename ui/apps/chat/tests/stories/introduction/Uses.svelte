<!--
	What this Storybook is for, as four tiles: designing with the real system, building from the components, checking with
	the accessibility addon (its findings are shown, not yet enforced; see a11y in .storybook/preview.ts) and handing over
	to agents. Each has its icon, its name and one line.
-->
<script lang="ts">
	import { Blocks, CircleCheck, PenTool, Send } from '@lucide/svelte';

	const uses = [
		{
			icon: PenTool,
			line: 'Try a change on the real components and tokens, in both themes',
			name: 'Designing'
		},
		{
			icon: Blocks,
			line: 'Look up a component, its states and tokens, light and dark',
			name: 'Building'
		},
		{
			icon: CircleCheck,
			line: 'Every story is the real code; the Accessibility panel lists what it misses',
			name: 'Checking'
		},
		{
			icon: Send,
			line: 'The rules are written for agents in DESIGN.md, its tokens generated from the CSS',
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
