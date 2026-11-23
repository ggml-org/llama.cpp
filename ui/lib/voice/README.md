# @ggml-org/llama-voice

Llama Voice, the voice input of Llama Chat, as a package: it records the microphone, shows its level as it goes, and
turns speech into text through llama-server, live while the user speaks and once more over the whole recording when
they stop. It also writes dictated text into a text field around what the user types. It has no UI and no
dependencies; Llama UI (`@ggml-org/llama-ui/voice`) draws the voice parts. It lives in llama.cpp at `ui/lib/voice`.

## Use

```ts
import { createDictation, startVoiceSession } from "@ggml-org/llama-voice";

const server = {
  resolveUrl: (path: string) => new URL(path, "http://localhost:8080/").toString(),
};

// dictate into a textarea: live text at the caret, then the whole recording's text in its place
const dictation = createDictation({
  getValue: () => textarea.value,
  setValue: (value) => (textarea.value = value),
  getCaretOffset: () => textarea.selectionStart,
  setCaretOffset: (offset) => textarea.setSelectionRange(offset, offset),
});

dictation.begin();

const session = await startVoiceSession({
  capture: { boost: true },
  realtime: true,
  transcribe: { context: server },
  onText: dictation.write,
});

textarea.oninput = dictation.edited;

// later
await dictation.finish(await session.stop());
```

In a Svelte app, `VoiceRecorder` from `@ggml-org/llama-voice/svelte` holds the same session as runes: `levels`,
`liveText`, `quiet` and `status` for a component to show.

## What is in it

- **Capture:** the microphone (`VoiceCapture`, `listMicrophones`, `pickMicrophone`), live levels, 16 kHz PCM chunks,
  and a WAV clip with its peaks when it stops.
- **Signal:** a `SpeechDetector` that tells speech from the room, the pause detector built on it, and a gain that
  raises quiet speech for transcription and says when it stays too quiet.
- **Transcription:** llama-server's realtime socket (`/v1/realtime`), with `/v1/audio/transcriptions` piece by piece
  when the socket is not there, and `transcribeClip` for a finished recording.
- **Dictation:** `createDictation` and the `DictationSpan` under it.
- **Support:** whether the browser can record, and why not, in words for the user (`VOICE_MICROPHONE_MESSAGE`, in
  English).

## What the app keeps

Llama Chat (`ui/apps/chat`) is the first app on it. It keeps:

- **The server and settings:** `VoiceService` turns its Voice settings and active server into the session's options.
- **Which model transcribes:** `resolveVoiceMode`, from the models the server has.
- **Its features:** the chat form's dictation and voice messages, and voice mode.

## Develop

```sh
npm ci
npm run check     # type check
npm test          # unit tests
npm run prepack   # build dist/ and lint the package
```

Llama Chat does not need a build of this package: it compiles `src/` directly, through the `@ggml-org/llama-voice`
aliases in its `vite.config.ts` and paths in its `tsconfig.json`. Its client tests cover what needs a browser
(recording, the recorder in a component).
