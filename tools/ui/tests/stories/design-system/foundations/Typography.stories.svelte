<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// the type roles from src/lib/styles/tokens/semantic.css: each sets size, line height and weight
	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Foundations/Typography'
	});

	const roles = [
		{ name: 'text-heading', spec: '24/32 600', use: 'rare page headings' },
		{ name: 'text-title', spec: '18/28 600', use: 'dialog and section titles' },
		{ name: 'text-body', spec: '16/24 400', use: 'chat messages and the chat input' },
		{ name: 'text-body-sm', spec: '14/20 400', use: 'most text, fields and lists' },
		{ name: 'text-label', spec: '14/20 500', use: 'buttons and field labels' },
		{ name: 'text-label-sm', spec: '12/16 500', use: 'badges and compact labels' },
		{ name: 'text-caption', spec: '12/16 400', use: 'tooltips, hints and metadata' }
	];
</script>

<Story asChild name="Default">
	<div class="w-[min(52rem,calc(100vw-2rem))] space-y-6 p-4">
		{#each roles as role (role.name)}
			<div class="grid grid-cols-[9rem_1fr] items-baseline gap-6">
				<div class="flex flex-col gap-0.5 font-mono text-[11px] text-muted-foreground">
					<span>{role.name}</span>

					<span class="opacity-70">{role.spec}</span>
				</div>

				<div class="flex flex-col gap-1">
					<span class={role.name}>The quick brown llama runs locally</span>

					<span class="text-caption text-fg-muted">{role.use}</span>
				</div>
			</div>
		{/each}

		<div class="grid grid-cols-[9rem_1fr] items-baseline gap-6">
			<span class="font-mono text-[11px] text-muted-foreground">text-fg-muted</span>

			<span class="text-body-sm text-fg-muted">Secondary text and hints</span>
		</div>

		<div class="grid grid-cols-[9rem_1fr] items-baseline gap-6">
			<span class="font-mono text-[11px] text-muted-foreground">font-mono</span>

			<code class="font-mono text-body-sm">llama-server -m model.gguf --port 8080</code>
		</div>

		<div class="grid grid-cols-[9rem_1fr] items-baseline gap-6">
			<span class="font-mono text-[11px] text-muted-foreground">engraved</span>

			<span class="text-body text-fg-muted engraved">Ask anything...</span>
		</div>
	</div>
</Story>
