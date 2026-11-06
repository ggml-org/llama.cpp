<!--
	Generated assets: the sources (the icon and the fonts) and a few of the files scripts build from them, shown as the
	real files from Brand › Assets: a lockup, app icons, a profile picture and a header
-->
<script lang="ts">
	import macos from '../../brand/assets/files/logo/svg/app/apple/macos-1024x1024-dark.svg?url';
	import pwa from '../../brand/assets/files/logo/svg/app/pwa/pwa-512x512-dark.svg?url';
	import windows from '../../brand/assets/files/logo/svg/app/windows/windows-app-icon-256x256-light.svg?url';
	import icon from '../../brand/assets/files/logo/svg/icon/icon-light.svg?url';
	import lockup from '../../brand/assets/files/logo/svg/lockup/llama-cpp/logo-llama-cpp-light.svg?url';
	import avatar from '../../brand/assets/files/logo/svg/social/x/x-avatar-dark.svg?url';
	import header from '../../brand/assets/files/logo/svg/social/x/x-header-llama-dark.svg?url';
</script>

<div class="assets">
	<div class="sources">
		<div class="tile rounded-card elevated-surface">
			<img alt="The Llama icon" src={icon} />
		</div>

		<div class="tile rounded-card elevated-surface">
			<span class="type">Aa</span>
		</div>

		<p class="note">Sources: the icon, Llama Sans and Llama Mono</p>
	</div>

	<p aria-hidden="true" class="arrow">→</p>

	<div class="outputs">
		<figure class="wide">
			<div class="tile rounded-card elevated-surface">
				<img alt="llama.cpp lockup" class="lockup" src={lockup} />
			</div>

			<figcaption>Lockup</figcaption>
		</figure>

		<figure>
			<div class="tile icons rounded-card elevated-surface">
				<img alt="macOS app icon" src={macos} />

				<img alt="Windows app icon" src={windows} />

				<img alt="Web app icon" src={pwa} />
			</div>

			<figcaption>App icons: macOS, Windows, web</figcaption>
		</figure>

		<figure>
			<div class="tile rounded-card elevated-surface">
				<img alt="Profile picture" class="round" src={avatar} />
			</div>

			<figcaption>Profile picture</figcaption>
		</figure>

		<figure class="wide">
			<div class="tile bleed rounded-card elevated-surface">
				<img alt="X header with the Llama lockup" src={header} />
			</div>

			<figcaption>X header, 1500 × 500, dark</figcaption>
		</figure>
	</div>
</div>

<style>
	.assets {
		display: grid;
		grid-template-columns: 9rem auto minmax(0, 1fr);
		gap: 20px;
		align-items: center;
	}

	.sources,
	.outputs {
		display: grid;
		gap: 12px;
	}

	.outputs {
		grid-template-columns: repeat(2, minmax(0, 1fr));
	}

	.wide {
		grid-column: span 2;
	}

	figure {
		display: grid;
		gap: 6px;
		margin: 0;
	}

	.tile {
		display: grid;
		place-items: center;
		height: 96px;
		padding: 16px;
		overflow: hidden;
		background: #fafafa;
	}

	.tile img {
		max-width: 100%;
		max-height: 100%;
	}

	.lockup {
		height: 40px;
	}

	.bleed {
		height: auto;
		aspect-ratio: 3;
		padding: 0;
	}

	.bleed img {
		width: 100%;
		height: 100%;
		object-fit: cover;
	}

	.icons {
		grid-auto-flow: column;
		gap: 12px;
	}

	.icons img {
		height: 56px;
	}

	.round {
		height: 64px;
		border-radius: 50%;
	}

	.sources .tile img {
		height: 48px;
	}

	.type {
		color: #111;
		font-size: 40px;
		font-weight: 600;
		line-height: 1;
	}

	.arrow {
		margin: 0;
		color: var(--figure-muted);
		font-size: 20px;
	}

	.note,
	figcaption {
		margin: 0;
		color: var(--figure-muted);
		font: 500 11px var(--mono);
	}

	@media (width < 720px) {
		.assets {
			grid-template-columns: minmax(0, 1fr);
		}

		.sources {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}

		.sources .note {
			grid-column: span 2;
		}

		.arrow {
			transform: rotate(90deg);
			justify-self: center;
		}
	}
</style>
