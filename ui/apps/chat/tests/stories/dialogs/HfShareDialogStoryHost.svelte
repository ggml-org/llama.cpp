<script lang="ts">
	import { asHfAccount, type FakeAccount } from '../fixtures/hf-fake-account.svelte.ts';
	import type { HfTraceFile } from '@ggml-org/llama-huggingface';
	import { HfShareDialog } from '@ggml-org/llama-huggingface/svelte';

	// the share dialog, open, on an account that calls out of nowhere: what its stories and its tests share

	interface Props {
		account: FakeAccount;
	}

	const SCOPES = [
		{ id: 'conversation', label: 'This conversation' },
		{ id: 'branch', label: 'This branch only' }
	];
	const TRACE: HfTraceFile = {
		commitMessage: 'Add trace',
		content: '{"type":"user","message":{"role":"user","content":"Hello"}}',
		path: 'trace.jsonl'
	};

	let { account }: Props = $props();

	let open = $state(true);
</script>

<HfShareDialog
	bind:open
	account={asHfAccount(account)}
	buildTrace={async () => TRACE}
	onCopyLink={() => {}}
	scopes={SCOPES}
	subject="this conversation"
/>
