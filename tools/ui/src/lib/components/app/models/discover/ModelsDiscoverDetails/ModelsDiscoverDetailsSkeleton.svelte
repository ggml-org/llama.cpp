<script lang="ts">
	import { Skeleton } from '$lib/components/ui/skeleton';
</script>

<!-- Static skeleton of ModelsDiscoverDetails: header, metadata chips, download options and readme lines. -->
<div class="@container space-y-6 p-6 pr-0" data-slot="model-details-skeleton">
	<!-- Header: avatar, name with capability icons, base model line, HF link -->
	<div class="space-y-3">
		<div class="flex items-start justify-between gap-3">
			<div class="flex min-w-0 items-center gap-2">
				<Skeleton class="h-12 w-12 rounded-md" />

				<div class="min-w-0 space-y-1.5">
					<Skeleton class="h-5 w-56 max-w-full" />

					<Skeleton class="h-3 w-40 max-w-full" />
				</div>
			</div>

			<Skeleton class="h-7.5 w-44 shrink-0 rounded-md" />
		</div>

		<!-- downloads / likes / last updated -->
		<div class="flex items-center gap-4">
			<Skeleton class="h-3.5 w-16" />

			<Skeleton class="h-3.5 w-12" />

			<Skeleton class="h-3.5 w-24" />
		</div>

		<!-- catalog family description, else the HF card description -->
		<Skeleton class="h-3.5 my-4 w-full" />

		<!-- metadata chips -->
		<div class="space-y-3">
			<div class="flex flex-wrap items-center gap-1.5">
				<Skeleton class="h-7 w-36 rounded-md" />

				<Skeleton class="h-7 w-28 rounded-md" />

				<Skeleton class="h-7 w-32 rounded-md" />

				<Skeleton class="h-7 w-24 rounded-md" />
			</div>
		</div>
	</div>

	<!-- Download options: quant rows and the draft opt-in line -->
	<div class="rounded-3xl border border-border/30 bg-muted/60 shadow-sm dark:border-border/20">
		<div class="flex w-full flex-col divide-y divide-border/50 px-4 pb-1 dark:divide-border/35">
			{#each [0, 1, 2] as _, index (index)}
				<div
					class="grid grid-cols-[5rem_1fr] items-center gap-3 py-3 @max-lg:grid-cols-1 @max-lg:gap-1.5"
				>
					<div class="space-y-1.5">
						<Skeleton class="h-4 w-14" />

						<Skeleton class="h-2.5 w-24" />
					</div>

					<div class="flex flex-wrap items-center justify-end gap-1.5">
						<Skeleton class="h-6 w-28 rounded-md" />

						<Skeleton class="h-6 w-20 rounded-md" />
					</div>
				</div>
			{/each}
		</div>

		<div class="space-y-2.5 border-t border-border/50 px-4 pt-3.5 dark:border-border/35">
			<Skeleton class="h-9 w-full rounded-md" />

			<div class="flex items-center gap-3">
				<span class="h-px flex-1 bg-border/50"></span>

				<Skeleton class="h-3 w-44" />

				<span class="h-px flex-1 bg-border/50"></span>
			</div>

			<Skeleton class="h-11 w-full rounded-md" />
		</div>
	</div>

	<!-- Readme -->
	<div class="p-3 rounded-3xl border border-border/30 bg-muted/60 shadow-sm dark:border-border/20">
		<div class="space-y-2.5">
			<Skeleton class="h-3.5 w-full" />

			<Skeleton class="h-3.5 w-full" />

			<Skeleton class="h-3.5 w-5/6" />

			<Skeleton class="h-3.5 w-2/3" />
		</div>
	</div>
</div>
