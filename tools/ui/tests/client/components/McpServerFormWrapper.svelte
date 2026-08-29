<script lang="ts">
	import McpServerForm from '$lib/components/app/mcp/McpServerForm.svelte';
	import { untrack } from 'svelte';

	interface Props {
		headers?: string;
	}

	let { headers = '' }: Props = $props();

	let headersState = $state(untrack(() => headers));
	let lastCapturedHeaders = $state(untrack(() => headers));

	$effect(() => {
		if (headers !== lastCapturedHeaders) {
			headersState = headers;
			lastCapturedHeaders = headers;
		}
	});
</script>

<!--
	Drives McpServerForm with a controlled `headers` string and exposes the
	latest captured value through `data-captured-headers` so the client test
	can read it back without a custom binding API.
-->
<McpServerForm
	headers={headersState}
	id="mcp-server-form-test"
	onHeadersChange={(value) => {
		headersState = value;
	}}
	onUrlChange={() => {}}
	url="https://example.test/mcp"
/>

<div data-captured-headers={headersState} data-testid="captured-headers" hidden></div>
