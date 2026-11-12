<script lang="ts">
	import { Badge, Button, Switch } from '#lib/index.js';
	import { setMode, userPrefersMode } from 'mode-watcher';

	const next = { light: 'dark', dark: 'system', system: 'light' } as const;
</script>

<!-- a quick check that the package's theme and components work in a SvelteKit app; Storybook is the showcase -->
<div class="flex min-h-screen flex-col items-center justify-center gap-6 p-6">
	<h1 class="text-heading">@ggml/llama-ui</h1>

	<p class="text-body-sm text-muted-foreground">The full set is in Storybook: <code>npm run storybook</code></p>

	<div class="flex flex-wrap items-center justify-center gap-3">
		<Button>Default</Button>
		<Button variant="secondary">Secondary</Button>
		<Button variant="outline">Outline</Button>
		<Badge variant="success">Downloaded</Badge>
		<Switch aria-label="Example switch" />
		<Button variant="ghost" onclick={() => setMode(next[userPrefersMode.current])}>
			Mode: {userPrefersMode.current}
		</Button>
	</div>
</div>
