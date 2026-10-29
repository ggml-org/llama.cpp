<script lang="ts" module>
	import {
		componentGroups,
		primitiveGroups,
		reference,
		semanticGroups,
		type TokenGroup
	} from './tokens';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// read from src/lib/styles/tokens/*.css, so the swatches follow the CSS as it changes
	const { Story } = defineMeta({
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Foundations/Colors'
	});
</script>

{#snippet tier(groups: TokenGroup[], intro: string, showReference: boolean)}
	<div class="w-[min(72rem,calc(100vw-2rem))] space-y-8 p-4">
		<p class="max-w-2xl text-body-sm text-fg-muted">{intro}</p>

		{#each groups as group (group.title)}
			<section>
				<h2 class="mb-3 text-label text-fg-muted first-letter:uppercase">{group.title}</h2>

				<div class="grid grid-cols-[repeat(auto-fill,minmax(11rem,1fr))] gap-4">
					{#each group.tokens as token (token.name)}
						<figure class="m-0 flex flex-col gap-2">
							<div
								style:background-color="var(--{token.name})"
								class="h-14 rounded-control border border-line-default"
							></div>

							<figcaption class="flex flex-col gap-0.5 font-mono text-[11px] leading-4">
								<span class="text-fg-default">--{token.name}</span>

								{#if showReference}
									<span class="truncate text-fg-muted" title={token.light}
										>light: {reference(token.light)}</span
									>

									{#if token.dark}
										<span class="truncate text-fg-muted" title={token.dark}
											>dark: {reference(token.dark)}</span
										>
									{/if}
								{:else}
									<span class="truncate text-fg-muted" title={token.light}>{token.light}</span>
								{/if}
							</figcaption>
						</figure>
					{/each}
				</div>
			</section>
		{/each}
	</div>
{/snippet}

<Story asChild name="Primitives">
	{@render tier(
		primitiveGroups,
		'Raw values named by what they are. Only semantic tokens point at these; components never use them directly.',
		false
	)}
</Story>

<Story asChild name="Semantic">
	{@render tier(
		semanticGroups,
		'Named by purpose. Each points at a primitive, and this is the only tier that changes between light and dark. Switch the theme to see the dark values.',
		true
	)}
</Story>

<Story asChild name="Components">
	{@render tier(
		componentGroups,
		'One group per component. They point at semantic tokens, so both themes follow automatically.',
		true
	)}
</Story>
