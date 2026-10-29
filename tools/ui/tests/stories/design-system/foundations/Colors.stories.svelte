<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';

	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Foundations/Colors'
	});

	// token pairs as defined on :root and .dark in src/app.css
	const groups: { title: string; tokens: string[] }[] = [
		{
			title: 'Surfaces',
			tokens: ['background', 'card', 'popover', 'muted', 'accent', 'secondary', 'code-background']
		},
		{
			title: 'Text',
			tokens: ['foreground', 'muted-foreground', 'card-foreground', 'code-foreground']
		},
		{ title: 'Brand and intent', tokens: ['primary', 'primary-foreground', 'destructive', 'ring'] },
		{ title: 'Lines', tokens: ['border', 'input'] },
		{
			title: 'Status',
			tokens: [
				'status-red',
				'status-red-ink',
				'status-yellow',
				'status-yellow-ink',
				'status-green',
				'status-green-ink'
			]
		},
		{ title: 'Charts', tokens: ['chart-1', 'chart-2', 'chart-3', 'chart-4', 'chart-5'] },
		{
			title: 'Sidebar',
			tokens: [
				'sidebar',
				'sidebar-foreground',
				'sidebar-primary',
				'sidebar-accent',
				'sidebar-border'
			]
		}
	];
</script>

<Story asChild name="Default">
	<div class="w-[min(72rem,calc(100vw-2rem))] space-y-8 p-4">
		{#each groups as group (group.title)}
			<section>
				<h2 class="mb-3 text-sm font-medium text-muted-foreground">{group.title}</h2>

				<div class="grid grid-cols-[repeat(auto-fill,minmax(9rem,1fr))] gap-4">
					{#each group.tokens as token (token)}
						<figure class="m-0 flex flex-col gap-2">
							<div
								style:background-color="var(--{token})"
								class="h-14 rounded-lg border border-border/60"
							></div>

							<figcaption class="font-mono text-[11px] text-muted-foreground">--{token}</figcaption>
						</figure>
					{/each}
				</div>
			</section>
		{/each}
	</div>
</Story>
