<!--
	What kind of model with audio input a model is, as a small badge beside its name in the model picker and the Models
	manager: a speech-to-text model that is for dictation rather than chat, or a chat model that hears and gets voice
	messages. A model that cannot hear shows nothing. The tooltip says what the kind means. Which model transcribes
	dictation is shown by the microphone among the row's actions instead (ModelRowActions).
-->
<script lang="ts">
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import { AudioLines, Captions } from '@lucide/svelte';
	import { VOICE_MODEL_ROLE_DESCRIPTION, VOICE_MODEL_ROLE_LABEL } from '#lib/constants/index.ts';
	import { VoiceModelRole } from '#lib/enums/index.ts';
	import { voiceStore } from '#lib/stores/index.ts';

	interface Props {
		model: string;
		class?: string;
	}

	let { class: className = '', model }: Props = $props();

	const ICONS = {
		[VoiceModelRole.TRANSCRIPTION]: Captions,
		[VoiceModelRole.VOICE]: AudioLines
	};

	let role = $derived(voiceStore.roleOf(model));
	let description = $derived(
		role === VoiceModelRole.TRANSCRIPTION
			? VOICE_MODEL_ROLE_DESCRIPTION.transcription
			: VOICE_MODEL_ROLE_DESCRIPTION.voice
	);
</script>

{#if role}
	{@const Icon = ICONS[role]}

	<Tooltip.Root>
		<Tooltip.Trigger>
			{#snippet child({ props })}
				<span
					{...props}
					class={[
						'inline-flex h-4 shrink-0 items-center gap-0.5 rounded-md border border-border/60 px-1 text-caption-sm leading-none font-medium whitespace-nowrap text-muted-foreground',
						className
					]}
					data-role={role}
					data-slot="model-voice-role"
				>
					<Icon aria-hidden="true" class="size-2.5" />

					{VOICE_MODEL_ROLE_LABEL[role]}
				</span>
			{/snippet}
		</Tooltip.Trigger>

		<Tooltip.Content class="max-w-64">
			<p>{description}</p>
		</Tooltip.Content>
	</Tooltip.Root>
{/if}
