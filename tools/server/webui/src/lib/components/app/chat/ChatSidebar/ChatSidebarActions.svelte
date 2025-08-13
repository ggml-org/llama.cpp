<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Search, SquarePen, X } from '@lucide/svelte';
    
    interface Props {
        handleMobileSidebarItemClick: () => void;
        isSearchModeActive: boolean;
        searchQuery: string;
    }

    let { handleMobileSidebarItemClick, isSearchModeActive = $bindable(), searchQuery = $bindable() }: Props = $props();

    let searchInput: HTMLInputElement | null = $state(null);

    $effect(() => {
        if (isSearchModeActive) {
            searchInput?.focus();
        }
    })

    function handleSearchModeDeactivate() {
        isSearchModeActive = false;
        searchQuery = '';
    }
</script>

<div class="space-y-0.5">
    {#if isSearchModeActive}
        <div class="relative">
            <Search class="text-muted-foreground absolute left-2 top-2.5 h-4 w-4" />

            <Input
                bind:ref={searchInput}
                onkeydown={(e) => e.key === 'Escape' && handleSearchModeDeactivate()}
                placeholder="Search conversations..."
                class="pl-8"
                bind:value={searchQuery}
            />

            <X
                class="cursor-pointertext-muted-foreground absolute right-2 top-2.5 h-4 w-4"
                onclick={handleSearchModeDeactivate}
            />
        </div>
    {:else}
        <Button
            class="w-full justify-start gap-2"
            href="/?new_chat=true"
            variant="ghost"
            onclick={handleMobileSidebarItemClick}
        >
            <SquarePen class="h-4 w-4" />

            New chat
        </Button>

        <Button
            class="w-full justify-start gap-2"
            variant="ghost"
            onclick={() => {
                isSearchModeActive = true;
            }}
        >
            <Search class="h-4 w-4" />

            Search conversations
        </Button>
    {/if}
</div>