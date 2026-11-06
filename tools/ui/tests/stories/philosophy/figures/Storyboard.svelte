<!--
	Simple by default, as a storyboard of the first run with Llama for Mac, in four frames built from the real pieces:
	the app's menu (Communications › Store's rebuild of it) at its first launch, with nothing installed and models
	suggested for this Mac, then downloading the recommended one; Llama Chat's first exchange; and Llama Chat's real
	settings, open at sampling, for when someone asks. The recommended model is not named: its row carries the Llama
	icon. The story seeds the chat stores (seedChatStory) so the settings and messages can render.
-->
<script lang="ts">
	import gpt from '../../brand/communications/store/logos/gpt.svg?raw';
	import qwen from '../../brand/communications/store/logos/qwen.svg?raw';
	import MacMenu, { type MacMenuRow } from '../../brand/communications/store/MacMenu.svelte';
	import { PATHS, SIZE } from '../../brand/logo-showcase/lib/logo';
	import { assistantMessage, noopChatActions, userMessage } from '../../chat/messages/fixtures';
	import ChatMessage from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessage.svelte';
	import SettingsChat from '$lib/components/app/settings/SettingsChat/SettingsChat.svelte';
	import { SETTINGS_SECTIONS } from '$lib/constants';

	const icon = `<svg viewBox="0 0 ${SIZE} ${SIZE}" fill="currentColor">${PATHS.map((d) => `<path d="${d}"/>`).join('')}</svg>`;

	const recommended: MacMenuRow[] = [
		{
			logo: icon,
			name: 'Recommended',
			params: '2B',
			subtitle: 'Reads, sees and hears  ∣  1.8 GB',
			vision: true
		},
		{ logo: gpt, name: 'gpt-oss', params: '20B', quant: 'MXFP4', subtitle: '12.1 GB' },
		{
			logo: qwen,
			name: 'Qwen3.8',
			params: '27B',
			quant: 'Q4_K_M',
			subtitle: '16.8 GB',
			vision: true
		}
	];
</script>

<ol class="storyboard">
	<li class="frame card-surface rounded-card">
		<p class="step first">1 · First launch</p>

		<div class="stage mac rounded-panel">
			<div class="menu"><MacMenu installed={[]} {recommended} /></div>
		</div>

		<p class="note">
			Nothing is installed. The menu suggests models that fit this Mac, and recommends one.
		</p>
	</li>

	<li class="frame card-surface rounded-card">
		<p class="step">2 · One click</p>

		<div class="stage mac rounded-panel">
			<div class="menu"><MacMenu downloading={0.62} installed={[]} {recommended} /></div>
		</div>

		<p class="note">
			The recommended model downloads from Hugging Face, and the server starts on its own.
		</p>
	</li>

	<li class="frame card-surface rounded-card">
		<p class="step">3 · First chat</p>

		<div class="stage chat rounded-panel">
			<div class="scaled">
				<ChatMessage chatActions={noopChatActions} message={userMessage} />

				<ChatMessage chatActions={noopChatActions} message={assistantMessage} />
			</div>
		</div>

		<p class="note">Open chat, and Llama Chat starts on a conversation, not on settings.</p>
	</li>

	<li class="frame card-surface rounded-card">
		<p class="step">4 · When you ask</p>

		<div class="stage settings rounded-panel" inert>
			<div class="scaled">
				<SettingsChat initialSection={SETTINGS_SECTIONS.SAMPLING_PENALTIES.slug} />
			</div>
		</div>

		<p class="note">Sampling, penalties and every other setting, one step away in Settings.</p>
	</li>
</ol>

<style>
	.storyboard {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 8px;
		margin: 0;
		padding: 0;
		list-style: none;
	}

	.frame {
		display: grid;
		grid-template-rows: auto 1fr auto;
		gap: 10px;
		margin: 0;
		padding: 14px;
	}

	p {
		margin: 0;
	}

	.step {
		color: var(--figure-muted);
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.step.first {
		color: var(--art-cinder);
	}

	.note {
		color: var(--figure-muted);
		font-size: 13px;
		line-height: 1.45;
	}

	/* every frame the same height, its piece cropped to it */
	.stage {
		height: 300px;
		overflow: hidden;
	}

	/* the menu as it drops from the menu bar, on a plain desktop */
	.mac {
		display: grid;
		justify-items: center;
		padding-top: 16px;
		background: linear-gradient(160deg, #c4ddf6, #dfecf9);
	}

	.menu {
		zoom: 0.9;
	}

	.chat,
	.settings {
		background: var(--surface-page);
	}

	.scaled {
		display: grid;
		gap: 16px;
		padding: 20px;
		zoom: 0.625;
	}

	@media (width < 720px) {
		.storyboard {
			grid-template-columns: minmax(0, 1fr);
		}
	}
</style>
