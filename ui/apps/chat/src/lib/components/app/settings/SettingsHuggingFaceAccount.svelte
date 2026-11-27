<script lang="ts">
	import { HfAccountPanel, HfLogo } from '@ggml-org/llama-huggingface/svelte';
	import { InsetList } from '@ggml-org/llama-ui/app/sheets';
	import { HUGGING_FACE_ENABLED } from '#lib/constants/index.ts';
	import { huggingFace } from '#lib/stores/index.ts';

	interface Props {
		/** As a group of a phone's settings sheet, where the desktop settings show a labelled row. */
		inset?: boolean;
	}

	let { inset = false }: Props = $props();

	const TITLE = 'Hugging Face';
	const CAPTION =
		"Conversations you share go to this account's datasets. The sign-in stays in this browser.";
</script>

<!-- a share is what signs the app in, so there is nothing to show before one did -->
{#if HUGGING_FACE_ENABLED && huggingFace.user}
	{#if inset}
		<InsetList caption={CAPTION} title={TITLE}>
			<div class="px-4 py-2">
				<HfAccountPanel account={huggingFace} />
			</div>
		</InsetList>
	{:else}
		<section aria-labelledby="settings-hugging-face-account" class="flex flex-col gap-2">
			<h3 class="flex items-center gap-1.5 text-label" id="settings-hugging-face-account">
				<HfLogo class="size-icon-md" />

				{TITLE}
			</h3>

			<HfAccountPanel account={huggingFace} />

			<p class="text-caption text-muted-foreground">{CAPTION}</p>
		</section>
	{/if}
{/if}
