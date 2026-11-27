/**
 * voiceTranscriptsStore - Transcripts of voice messages that are on their way
 *
 * A voice message is attached as soon as the recording stops, while its transcript
 * can take a few seconds more. This keeps each pending transcript by the clip's name
 * (unique per recording), so the chip in the form and the sent message show its
 * state, a send picks it up once it is there, and a message sent before it arrived
 * gets it written into its stored attachment afterwards. Kept in memory only.
 */

import { VoiceTranscriptStatus } from '@ggml-org/llama-ui/voice';
import { VOICE_TRANSCRIPT_APPLY_RETRY_DELAYS_MS } from '#lib/constants/index.ts';
import { AttachmentType } from '#lib/enums/index.ts';
import { DatabaseService } from '#lib/services/database.service.ts';
import { VoiceService } from '#lib/services/voice.service.ts';
// direct imports between stores, not via the barrel, to avoid circular deps
import { conversationsStore } from '#lib/stores/conversations/index.svelte.ts';
import type {
	DatabaseMessage,
	DatabaseMessageExtra,
	DatabaseMessageExtraAudioFile
} from '#lib/types/index.ts';
import { SvelteMap } from 'svelte/reactivity';

export type VoiceTranscriptState =
	| { status: VoiceTranscriptStatus.TRANSCRIBING }
	| { status: VoiceTranscriptStatus.DONE; transcript: string }
	| {
			status: VoiceTranscriptStatus.ERROR;
			message: string;
			/** Why, in words for the user, when the app can tell (VoiceService.failureReason) */
			reason: string | null;
	  };

function isPendingClip(extra: DatabaseMessageExtra, name: string): boolean {
	return (
		extra.type === AttachmentType.AUDIO &&
		extra.name === name &&
		extra.voice === true &&
		extra.transcript === undefined
	);
}

/** The extras with the clip's transcript filled in, or null when none of them is that clip. */
export function withTranscript(
	extras: DatabaseMessageExtra[] | undefined,
	name: string,
	transcript: string
): DatabaseMessageExtra[] | null {
	if (!extras?.some((extra) => isPendingClip(extra, name))) return null;

	return extras.map((extra) =>
		isPendingClip(extra, name)
			? ({ ...(extra as DatabaseMessageExtraAudioFile), transcript } as DatabaseMessageExtra)
			: extra
	);
}

class VoiceTranscriptsStore {
	// clips whose transcript the user wrote while recording: the model's does not replace it
	private edited = new Set<string>();
	private entries = new SvelteMap<string, VoiceTranscriptState>();
	private recordedAt = new Map<string, number>();

	/** Drops a clip removed from the form before it was sent, or whose message has its transcript. */
	forget(name: string): void {
		this.entries.delete(name);
		this.recordedAt.delete(name);
		this.edited.delete(name);
	}

	/** The state of a clip's transcript, undefined when none was asked for in this session. */
	get(name: string): VoiceTranscriptState | undefined {
		return this.entries.get(name);
	}

	/** Sets a clip's transcript to the text the user edited while recording; the model's is ignored. */
	setTranscript(name: string, transcript: string): void {
		this.edited.add(name);
		this.recordedAt.set(name, this.recordedAt.get(name) ?? Date.now());
		this.entries.set(name, { status: VoiceTranscriptStatus.DONE, transcript });
	}

	/** A clip's transcript as it is shown: the one stored with it, else the one on its way. */
	shown(
		name: string,
		stored?: string
	): { status: VoiceTranscriptStatus; transcript?: string; errorReason?: string | null } {
		const transcript = stored ?? this.transcriptOf(name);

		if (transcript !== undefined) return { status: VoiceTranscriptStatus.DONE, transcript };

		const entry = this.entries.get(name);

		return {
			errorReason: entry?.status === VoiceTranscriptStatus.ERROR ? entry.reason : null,
			status: entry?.status ?? VoiceTranscriptStatus.IDLE
		};
	}

	/** Follows a clip's transcript, and writes it into the message it went out with. */
	track(name: string, transcript: Promise<string>): void {
		this.entries.set(name, { status: VoiceTranscriptStatus.TRANSCRIBING });
		this.recordedAt.set(name, Date.now());

		transcript.then(
			(text) => {
				if (!this.entries.has(name) || this.edited.has(name)) return;

				this.entries.set(name, { status: VoiceTranscriptStatus.DONE, transcript: text });
				void this.applyToMessage(name, text);
			},
			(error: unknown) => {
				if (!this.entries.has(name) || this.edited.has(name)) return;

				console.warn('Voice message transcription failed:', error);
				this.entries.set(name, {
					message: error instanceof Error ? error.message : String(error),
					reason: VoiceService.failureReason(error),
					status: VoiceTranscriptStatus.ERROR
				});
			}
		);
	}

	/** The transcript once it is there. */
	transcriptOf(name: string): string | undefined {
		const entry = this.entries.get(name);

		return entry?.status === VoiceTranscriptStatus.DONE ? entry.transcript : undefined;
	}

	private async applyToMessage(name: string, transcript: string): Promise<void> {
		for (const delay of [0, ...VOICE_TRANSCRIPT_APPLY_RETRY_DELAYS_MS]) {
			if (delay) await new Promise((resolve) => setTimeout(resolve, delay));

			// in its message now, which shows it from there
			if (await this.writeTranscript(name, transcript)) return this.forget(name);
		}
		// not sent (still in the form, or dropped): a send reads it from here
	}

	private async writeTranscript(name: string, transcript: string): Promise<boolean> {
		const active = (conversationsStore.activeMessages as DatabaseMessage[]).find((message) =>
			message.extra?.some((extra) => isPendingClip(extra, name))
		);

		if (active) {
			// plain objects: IndexedDB cannot clone the state proxies
			const extra = withTranscript($state.snapshot(active.extra), name, transcript)!;

			await DatabaseService.updateMessage(active.id, { extra });
			conversationsStore.updateMessageAtIndex(conversationsStore.findMessageIndex(active.id), {
				extra
			});

			return true;
		}

		// the chat was left before the transcript came back: the message is newer than the clip
		const since = this.recordedAt.get(name) ?? Date.now();
		const stored = (await DatabaseService.getMessagesSince(since)).find(
			(message: DatabaseMessage) => message.extra?.some((extra) => isPendingClip(extra, name))
		);

		if (!stored) return false;

		await DatabaseService.updateMessage(stored.id, {
			extra: withTranscript(stored.extra, name, transcript)!
		});

		return true;
	}
}

export const voiceTranscriptsStore = new VoiceTranscriptsStore();
