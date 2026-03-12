<script lang="ts">
	import * as Tooltip from '../src/lib/components/ui/tooltip';

	interface Props {
		children: any;
	}

	let { children }: Props = $props();
</script>

<Tooltip.Provider>
	{@render children()}
</Tooltip.Provider>
