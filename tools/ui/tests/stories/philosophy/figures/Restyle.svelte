<!--
	The same real controls three times: as Llama UI ships them, then made over as two well-known UIs draw theirs, as an
	app built on Llama UI could. The colours go through the component tokens; the sizes, corners and type, which the
	theme inlines into the components' classes, through a few rules on the components' slots. The markup stays the same.
	GitHub's values are Primer's, measured on github.com and primer.style: 32px buttons with 6px corners, the green
	primary and grey default button, the blue topic tag and ToggleSwitch, its knob a rounded square over half the track.
	macOS's are AppKit's on macOS 27, drawn by NSButton and NSSwitch: 24pt push buttons with 6pt corners at 13pt, the
	blue default button on a grey one, and the wide switch with its capsule knob
-->
<script lang="ts">
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import { Switch } from '$lib/components/ui/switch';

	const looks = [
		{ label: 'As shipped', look: '' },
		{ label: 'GitHub', look: 'github' },
		{ label: 'macOS', look: 'macos' }
	];
</script>

<div class="looks">
	{#each looks as { label, look } (label)}
		<div class={['look card-surface rounded-card', look]}>
			<div class="row">
				<span class="label">{label}</span>

				<Badge class="ml-auto">Q4_K_M</Badge>
			</div>

			<div class="row">
				<Button>Load model</Button>

				<Button variant="secondary">Settings</Button>
			</div>

			<div class="row">
				<Switch aria-label="Keep in memory" checked />

				<span aria-hidden="true" class="setting text-body-sm">Keep in memory</span>
			</div>
		</div>
	{/each}
</div>

<style>
	.looks {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
		gap: 16px;
	}

	.look {
		display: grid;
		gap: 16px;
		padding: 20px;
	}

	.label {
		color: var(--figure-muted);
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.row {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
		align-items: center;
	}

	/* GitHub: Primer's light values, then its dark ones */
	.github {
		--ink: #1f2328;
		--accent: #0969da;
		--card-background: #fff;
		--card-edge-side: #d1d9e0;
		--card-edge-mid: #d1d9e0;
		--card-highlight: transparent;
		--card-shadow: none;
		--key-sheen: transparent;
		--key-sheen-solid: transparent;
		--button-shadow: 0 1px 0 rgb(31 35 40 / 0.04);
		--button-text: #25292e;
		--button-primary-background: #1f883d;
		--button-primary-background-hover: #1c8139;
		--button-primary-edge: rgb(31 35 40 / 0.15);
		--button-primary-text: #fff;
		--button-secondary-background: #f6f8fa;
		--button-secondary-background-hover: #eff2f5;
		--button-secondary-border: #d1d9e0;
		--switch-track: #e6eaef;
		--switch-track-on: var(--accent);
		--switch-thumb: #fff;
		--switch-thumb-on: #fff;
		--knob-edge: #818b98;
		--badge-primary-background: #ddf4ff;
		--badge-primary-background-hover: #ddf4ff;
		--badge-primary-text: var(--accent);

		border-radius: 6px;
		color: var(--ink);
		font-family:
			-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
	}

	:global(.dark) .github {
		--ink: #f0f6fc;
		--accent: #1f6feb;
		--card-background: #0d1117;
		--card-edge-side: #3d444d;
		--card-edge-mid: #3d444d;
		--button-shadow: 0 0 #0000;
		--button-text: #f0f6fc;
		--button-primary-background: #238636;
		--button-primary-background-hover: #29903b;
		--button-primary-edge: rgb(255 255 255 / 0.15);
		--button-secondary-background: #212830;
		--button-secondary-background-hover: #262c36;
		--button-secondary-border: #3d444d;
		--switch-track: #2a313c;
		--switch-thumb: #010409;
		--switch-thumb-on: #010409;
		--knob-edge: #858f99;
		--badge-primary-background: rgb(56 139 253 / 0.1);
		--badge-primary-background-hover: rgb(56 139 253 / 0.1);
		--badge-primary-text: #4493f8;
	}

	.github :global([data-slot='button']) {
		height: 32px;
		padding: 0 12px;
		border-radius: 6px;
		font-size: 14px;
		font-weight: 500;
	}

	/* the topic tag: a pill in semibold */
	.github :global([data-slot='badge']) {
		padding: 2px 10px;
		border-radius: 999px;
		font-size: 12px;
		font-weight: 600;
		line-height: 18px;
	}

	/* ToggleSwitch: a 64 × 32 track with 6px corners, a knob with 4px corners over half of it, and on the free half
	   the line that reads as on */
	.github :global([data-slot='switch']) {
		position: relative;
		width: 64px;
		height: 32px;
		border-color: var(--switch-track);
		border-radius: 6px;
		box-shadow: none;
	}

	.github :global([data-slot='switch'][data-state='checked']) {
		border-color: var(--accent);
	}

	.github :global([data-slot='switch'][data-state='checked'])::before {
		position: absolute;
		top: 50%;
		left: 25%;
		width: 1.5px;
		height: 12px;
		border-radius: 1px;
		background: #fff;
		content: '';
		translate: -50% -50%;
	}

	.github :global([data-slot='switch-thumb']) {
		width: 50%;
		height: 100%;
		box-sizing: border-box;
		border: 1px solid var(--knob-edge);
		border-radius: 4px;
		box-shadow: none;
		translate: 0;
	}

	.github :global([data-slot='switch-thumb'][data-state='checked']) {
		border-color: var(--accent);
		translate: 100% 0;
	}

	.github .setting {
		font-size: 14px;
	}

	/* macOS: AppKit's light values, then its dark ones */
	.macos {
		--ink: rgb(0 0 0 / 0.85);
		--accent: #418ff8;
		--card-background: #fff;
		--card-edge-side: rgb(0 0 0 / 0.1);
		--card-edge-mid: rgb(0 0 0 / 0.1);
		--card-highlight: transparent;
		--card-shadow: 0 6px 20px rgb(0 0 0 / 0.08);
		--key-sheen: transparent;
		--key-sheen-solid: transparent;
		--button-shadow: 0 0 #0000;
		--button-text: var(--ink);
		--button-primary-background: var(--accent);
		--button-primary-background-hover: var(--accent);
		--button-primary-edge: transparent;
		--button-primary-text: #fff;
		--button-secondary-background: rgb(0 0 0 / 0.08);
		--button-secondary-background-hover: rgb(0 0 0 / 0.08);
		--button-secondary-border: transparent;
		--switch-track: rgb(0 0 0 / 0.09);
		--switch-track-on: #3f8cf4;
		--switch-thumb: #fff;
		--switch-thumb-on: #fff;
		--badge-primary-background: var(--accent);
		--badge-primary-background-hover: var(--accent);
		--badge-primary-text: #fff;

		border-radius: 16px;
		color: var(--ink);
		font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
	}

	:global(.dark) .macos {
		--ink: rgb(255 255 255 / 0.85);
		--card-background: #282828;
		--card-edge-side: rgb(255 255 255 / 0.1);
		--card-edge-mid: rgb(255 255 255 / 0.1);
		--card-shadow: 0 6px 20px rgb(0 0 0 / 0.3);
		--button-secondary-background: rgb(255 255 255 / 0.1);
		--button-secondary-background-hover: rgb(255 255 255 / 0.1);
		--switch-track: rgb(255 255 255 / 0.14);
		--switch-track-on: #4791f8;
	}

	.macos .row {
		gap: 12px;
	}

	.macos :global([data-slot='button']) {
		height: 24px;
		padding: 0 12px;
		border-radius: 6px;
		font-size: 13px;
		font-weight: 400;
	}

	.macos :global([data-slot='badge']) {
		padding: 1px 7px;
		border-radius: 999px;
		font-size: 11px;
		line-height: 14px;
	}

	/* NSSwitch at its regular size: a 54 × 24 capsule with a 32 × 20 capsule knob, 2pt in from the edge */
	.macos :global([data-slot='switch']) {
		width: 54px;
		height: 24px;
		padding: 0 2px;
		border: 0;
		box-shadow: none;
	}

	.macos :global([data-slot='switch-thumb']) {
		width: 32px;
		height: 20px;
		border-radius: 10px;
		box-shadow: 0 1px 2px rgb(0 0 0 / 0.12);
		translate: 0;
	}

	.macos :global([data-slot='switch-thumb'][data-state='checked']) {
		translate: 18px 0;
	}

	.macos .setting {
		font-size: 13px;
	}
</style>
