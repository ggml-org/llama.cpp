<!--
	A model's part in voice input, as a small badge beside its name in the model picker and the Models manager: the
	model that transcribes dictation (in the accent colour, as there is one), a speech-to-text model that is for dictation
	rather than chat, or a chat model that hears and gets voice messages. A model that cannot hear shows nothing. The
	tooltip says what the part means.
-->
<script lang="ts">
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import { AudioLines, Captions, Mic } from '@lucide/svelte';
	import { VOICE_MODEL_ROLE_DESCRIPTION, VOICE_MODEL_ROLE_LABEL } from '#lib/constants/index.ts';
	import { VoiceModelRole } from '#lib/enums/index.ts';
	import { voiceStore } from '#lib/stores/index.ts';

	interface Props {
		model: string;
		class?: string;
	}

	let { class: className = '', model }: Props = $props();

	const ICONS = {
		[VoiceModelRole.DICTATION]: Mic,
		[VoiceModelRole.TRANSCRIPTION]: Captions,
		[VoiceModelRole.VOICE]: AudioLines
	};

	let role = $derived(voiceStore.roleOf(model));
	let description = $derived.by(() => {
		switch (role) {
			case VoiceModelRole.DICTATION:
				return voiceStore.dictationIsAutomatic
					? VOICE_MODEL_ROLE_DESCRIPTION.dictationAuto
					: VOICE_MODEL_ROLE_DESCRIPTION.dictationChosen;
			case VoiceModelRole.TRANSCRIPTION:
				return VOICE_MODEL_ROLE_DESCRIPTION.transcription;
			default:
				return VOICE_MODEL_ROLE_DESCRIPTION.voice;
		}
	});
</script>

{#if role}
	{@const Icon = ICONS[role]}

	<Tooltip.Root>
		<Tooltip.Trigger>
			{#snippet child({ props })}
				<span
					{...props}
					class={[
						'inline-flex h-4 shrink-0 items-center gap-0.5 rounded-md px-1 text-[10px] leading-none font-medium whitespace-nowrap',
						role === VoiceModelRole.DICTATION
							? 'bg-primary text-primary-foreground'
							: 'border border-border/60 text-muted-foreground',
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
