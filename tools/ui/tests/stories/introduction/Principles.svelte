<!--
	How we think, as five tiles, each with a small picture of the idea made from the real system: the devices a model
	runs on, the first run's recommendation, the three places the character lives, a control on its grid, and one
	decision through the three token tiers.
-->
<script lang="ts">
	import { PATHS, SIZE } from '../brand/logo-showcase/lib/logo';
	import { Laptop, Lock, Monitor, Smartphone, UserX, WifiOff } from '@lucide/svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
</script>

<div class="principles">
	<section class="principle wide card-surface rounded-card">
		<div class="picture">
			<div class="devices">
				<Smartphone aria-hidden="true" size={28} />

				<Laptop aria-hidden="true" size={36} />

				<Monitor aria-hidden="true" size={36} />
			</div>

			<div class="chips">
				<Badge variant="outline"><WifiOff aria-hidden="true" />Offline</Badge>

				<Badge variant="outline"><UserX aria-hidden="true" />No account</Badge>

				<Badge variant="outline"><Lock aria-hidden="true" />Stays on the device</Badge>
			</div>
		</div>

		<h3>Personal, on hardware you already have</h3>

		<p>Private by default, and open to everyone.</p>
	</section>

	<section class="principle card-surface rounded-card">
		<div class="picture">
			<div class="pick rounded-panel bg-surface-lifted elevated-control">
				<Badge>Recommended</Badge>

				<span class="pick-note">Reads, sees and hears</span>

				<Button size="sm">Download</Button>
			</div>
		</div>

		<h3>Simple by default</h3>

		<p>Choices made for you to start; every setting when you want it.</p>
	</section>

	<section class="principle card-surface rounded-card">
		<div class="picture character">
			<svg aria-hidden="true" class="icon" viewBox="0 0 {SIZE} {SIZE}">
				{#each PATHS as d (d)}<path {d} />{/each}
			</svg>

			<span class="aa">Aa</span>

			<span aria-hidden="true" class="cinder"></span>
		</div>

		<h3>Recognisable, easy to change</h3>

		<p>The icon, the type, one accent. The rest is neutral.</p>
	</section>

	<section class="principle card-surface rounded-card">
		<div class="picture grid-bg rounded-panel">
			<Button size="sm" variant="outline">Run</Button>
		</div>

		<h3>Function first</h3>

		<p>Every element has a job and a reason for its size.</p>
	</section>

	<section class="principle card-surface rounded-card">
		<div class="picture chain">
			<code>--neutral-900</code>

			<span aria-hidden="true">↓</span>

			<code>--action-primary</code>

			<span aria-hidden="true">↓</span>

			<code>--button-primary-background</code>
		</div>

		<h3>Design lives in code</h3>

		<p>Change a primitive and the rest follows.</p>
	</section>
</div>

<style>
	.principles {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 8px;
	}

	.principle {
		display: grid;
		grid-template-rows: auto auto 1fr;
		gap: 6px;
		padding: 16px;
	}

	.wide {
		grid-column: span 2;
	}

	.picture {
		display: grid;
		place-items: center;
		min-height: 112px;
		margin-bottom: 8px;
	}

	.devices {
		display: flex;
		align-items: end;
		gap: 20px;
		color: var(--fg-default);
	}

	.wide .picture {
		gap: 14px;
	}

	.chips {
		display: flex;
		flex-wrap: wrap;
		justify-content: center;
		gap: 6px;
	}

	.pick {
		display: grid;
		justify-items: start;
		gap: 8px;
		padding: 12px 14px;
	}

	.pick-note {
		color: var(--figure-muted);
		font-size: 12px;
	}

	.character {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 16px;
	}

	.icon {
		width: 40px;
		height: 40px;
		fill: currentColor;
	}

	.aa {
		font-size: 40px;
		font-weight: 600;
		line-height: 1;
	}

	.cinder {
		width: 18px;
		height: 18px;
		border-radius: 50%;
		background: var(--art-cinder);
	}

	.grid-bg {
		background-image:
			linear-gradient(
				color-mix(in oklab, var(--line-default) 50%, transparent) 1px,
				transparent 1px
			),
			linear-gradient(
				90deg,
				color-mix(in oklab, var(--line-default) 50%, transparent) 1px,
				transparent 1px
			);
		background-position: center;
		background-size: 8px 8px;
	}

	.chain {
		gap: 2px;
		color: var(--figure-muted);
		font-size: 12px;
	}

	.chain code {
		color: var(--fg-default);
		font: 500 12px var(--mono);
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

	@media (width < 640px) {
		.principles {
			grid-template-columns: minmax(0, 1fr);
		}

		.wide {
			grid-column: auto;
		}
	}
</style>
