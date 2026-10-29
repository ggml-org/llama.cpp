<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';

	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Foundations/Typography'
	});

	const sizes = ['text-2xl', 'text-xl', 'text-lg', 'text-base', 'text-sm', 'text-xs'];
</script>

<Story asChild name="Default">
	<div class="w-[min(48rem,calc(100vw-2rem))] space-y-6 p-4">
		{#each sizes as size (size)}
			<div class="flex items-baseline gap-6">
				<span class="w-20 shrink-0 font-mono text-[11px] text-muted-foreground">{size}</span>

				<span class={size}>The quick brown llama runs locally</span>
			</div>
		{/each}

		<div class="flex items-baseline gap-6">
			<span class="w-20 shrink-0 font-mono text-[11px] text-muted-foreground">muted</span>

			<span class="text-sm text-muted-foreground">Secondary text and hints</span>
		</div>

		<div class="flex items-baseline gap-6">
			<span class="w-20 shrink-0 font-mono text-[11px] text-muted-foreground">font-mono</span>

			<code class="font-mono text-sm">llama-server -m model.gguf --port 8080</code>
		</div>
	</div>
</Story>
