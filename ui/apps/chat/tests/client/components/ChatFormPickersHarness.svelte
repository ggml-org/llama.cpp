<script lang="ts">
	import {
		useChatFormPickers,
		type UseChatFormPickersReturn
	} from '#lib/hooks/use-chat-form-pickers.svelte.ts';

	interface Props {
		cwd?: string | null;
		serverHome?: string | null;
	}

	let { cwd = null, serverHome = null }: Props = $props();

	let value = $state('');
	let caretOffset = $state(0);
	const calls: string[] = [];

	const pickers = useChatFormPickers({
		canCompact: () => true,
		editsCwdInline: () => true,
		focusInput: () => {},
		getCaretOffset: () => caretOffset,
		getCwd: () => cwd,
		getPickersRef: () => undefined,
		getServerHome: () => serverHome,
		getShowModelSelector: () => true,
		getValue: () => value,
		hasCwdTools: () => true,
		hasPrompts: () => true,
		onCompact: () => {
			calls.push('onCompact');
		},
		openModelSelector: () => {
			calls.push('openModelSelector');
		},
		opensOnInput: () => true,
		setCaretOffset: (o) => {
			caretOffset = o;
		},
		setValue: (v) => {
			value = v;
			calls.push(`setValue:${v}`);
		}
	});

	// Simulate the user typing: update the buffer and run the input flow.
	export function type(text: string) {
		value = text;
		caretOffset = text.length;
		pickers.handleInput();
	}

	export function getCaretOffset() {
		return caretOffset;
	}

	export function getValue() {
		return value;
	}

	export function getCalls() {
		return calls;
	}

	export function getPickers(): UseChatFormPickersReturn {
		return pickers;
	}
</script>
