<script lang="ts" module>
	import { SquareTerminal } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import CollapsibleTerminalBlock from '$lib/components/app/content/CollapsibleTerminalBlock.svelte';

	const { Story } = defineMeta({
		component: CollapsibleTerminalBlock,
		parameters: {
			layout: 'centered'
		},
		title: 'Content/CollapsibleTerminalBlock'
	});
</script>

<Story asChild name="Default">
	<div class="w-[36rem]">
		<CollapsibleTerminalBlock icon={SquareTerminal} open subtitle="exit 0" title="cargo test">
			<pre class="p-3 font-mono text-xs">running 3 tests
test borrow::longest ... ok
test borrow::shorter ... ok
test borrow::equal ... ok

test result: ok. 3 passed; 0 failed</pre>
		</CollapsibleTerminalBlock>
	</div>
</Story>
