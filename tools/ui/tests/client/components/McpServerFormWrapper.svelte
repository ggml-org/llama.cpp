<script lang="ts">
	import { untrack } from 'svelte';
	import McpServerForm from '$lib/components/app/mcp/McpServerForm.svelte';

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
	url="https://example.test/mcp"
	headers={headersState}
	onUrlChange={() => {}}
	onHeadersChange={(value) => {
		headersState = value;
	}}
	id="mcp-server-form-test"
/>

<div data-testid="captured-headers" data-captured-headers={headersState} hidden></div>
