<script lang="ts">
	import {
		useChatFormPickers,
		type UseChatFormPickersReturn
	} from '$lib/hooks/use-chat-form-pickers.svelte';

	let value = $state('');
	let caretOffset = $state(0);
	const calls: string[] = [];

	const pickers = useChatFormPickers({
		getValue: () => value,
		setValue: (v) => {
			value = v;
			calls.push(`setValue:${v}`);
		},
		getCaretOffset: () => caretOffset,
		setCaretOffset: (o) => {
			caretOffset = o;
		},
		focusInput: () => {},
		getShowModelSelector: () => true,
		hasPrompts: () => true,
		hasCwdTools: () => true,
		getCwd: () => null,
		getServerHome: () => null,
		openModelSelector: () => {
			calls.push('openModelSelector');
		},
		getPickersRef: () => undefined
	});

	// Simulate the user typing: update the buffer and run the input flow.
	export function type(text: string) {
		value = text;
		caretOffset = text.length;
		pickers.handleInput();
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
