/**
 * Voice UI: how a voice clip is shown while it is recorded, once it is recorded, and with its transcript. Every
 * component takes its data in (levels, peaks, an audio URL, a transcript); recording, analysing and transcribing stay
 * in each app.
 */
export {
  default as VoiceRecordingBar,
  type VoiceRecordingBarLabels,
} from "./VoiceRecordingBar.svelte";
export { default as VoiceSpokenText } from "./VoiceSpokenText.svelte";
export { default as VoiceTranscriptReveal } from "./VoiceTranscriptReveal.svelte";
export { default as VoiceWaveform } from "./VoiceWaveform.svelte";
export { default as VoiceWaveformLive } from "./VoiceWaveformLive.svelte";
export * from "../../voice/index.js";
