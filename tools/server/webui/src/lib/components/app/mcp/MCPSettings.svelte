<!-- [AI] MCP Settings UI component -->
<script lang="ts">
	import { Server, Plus, Trash2, Power, PowerOff } from '@lucide/svelte';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Switch } from '$lib/components/ui/switch';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { MCPServerStatus, type MCPServer } from '$lib/types';

	interface Props {
		onSave?: () => void;
	}

	let { onSave }: Props = $props();

	let newServer = $state({
		id: '',
		name: '',
		command: '',
		args: '',
		enabled: true
	});

	let isAdding = $state(false);

	async function handleAddServer() {
		console.log('handleAddServer called', { id: newServer.id, name: newServer.name, command: newServer.command, args: newServer.args });
		if (!newServer.id || !newServer.name || !newServer.command) {
			return;
		}

		try {
			await mcpStore.addServer({
				id: newServer.id,
				name: newServer.name,
				command: newServer.command,
			args: newServer.args
				.split(" ")
				.map(a => a.trim().replace(/^["']|["']$/g, ""))
						.filter((a) => a),
				enabled: newServer.enabled
			});

			// Reset form
			newServer = {
				id: '',
				name: '',
				command: '',
				args: '',
				enabled: true
			};
			isAdding = false;
		} catch (error) {
			console.error('Failed to add server:', error);
		}
	}

	async function handleToggleServer(server: MCPServer) {
		try {
			await mcpStore.updateServer(server.id, { enabled: !server.enabled });
		} catch (error) {
			console.error('Failed to toggle server:', error);
		}
	}

	async function handleRemoveServer(serverId: string) {
		if (!confirm('Are you sure you want to remove this server?')) {
			return;
		}

		try {
			await mcpStore.removeServer(serverId);
		} catch (error) {
			console.error('Failed to remove server:', error);
		}
	}

	function getStatusColor(status: MCPServerStatus): string {
		switch (status) {
			case MCPServerStatus.RUNNING:
				return 'text-green-500';
			case MCPServerStatus.STARTING:
				return 'text-yellow-500';
			case MCPServerStatus.ERROR:
				return 'text-red-500';
			default:
				return 'text-gray-500';
		}
	}

	function getStatusIcon(status: MCPServerStatus) {
		return status === MCPServerStatus.RUNNING ? Power : PowerOff;
	}
</script>

<div class="flex h-full flex-col">
	<!-- Header -->
	<div class="border-b p-6">
		<div class="flex items-center gap-3">
			<Server class="h-6 w-6" />
			<h2 class="text-2xl font-semibold">MCP Servers</h2>
		</div>
		<p class="mt-2 text-sm text-muted-foreground">
			Manage Model Context Protocol servers for tool calling
		</p>
	</div>

	<!-- Content -->
	<ScrollArea class="flex-1 p-6">
		<div class="space-y-4">
			<!-- Server List -->
			{#each mcpStore.servers as server (server.id)}
				{@const StatusIcon = getStatusIcon(server.status)}
				<div class="rounded-lg border p-4">
					<div class="flex items-start justify-between">
						<div class="flex-1">
							<div class="flex items-center gap-2">
								<h3 class="font-medium">{server.name}</h3>
								<StatusIcon class="h-4 w-4 {getStatusColor(server.status)}" />
								<span class="text-xs text-muted-foreground">{server.status}</span>
							</div>
							<p class="mt-1 text-sm text-muted-foreground">
								{server.command}
								{server.args.join(' ')}
							</p>
							<p class="mt-1 text-xs text-muted-foreground">ID: {server.id}</p>
						</div>

						<div class="flex items-center gap-2">
							<Switch
								checked={server.enabled}
								onCheckedChange={() => handleToggleServer(server)}
							/>
							<Button
								variant="ghost"
								size="icon"
								onclick={() => handleRemoveServer(server.id)}
							>
								<Trash2 class="h-4 w-4" />
							</Button>
						</div>
					</div>
				</div>
			{/each}

			{#if mcpStore.servers.length === 0 && !isAdding}
				<div class="rounded-lg border border-dashed p-8 text-center">
					<Server class="mx-auto h-12 w-12 text-muted-foreground" />
					<p class="mt-4 text-sm text-muted-foreground">No MCP servers configured</p>
					<Button class="mt-4" onclick={() => (isAdding = true)}>
						<Plus class="mr-2 h-4 w-4" />
						Add Server
					</Button>
				</div>
			{/if}

			<!-- Add Server Form -->
			{#if isAdding}
				<div class="rounded-lg border bg-muted/50 p-4">
					<h3 class="mb-4 font-medium">Add New Server</h3>
					<div class="space-y-3">
						<div>
							<Label for="server-id">Server ID</Label>
							<Input
								id="server-id"
								bind:value={newServer.id}
								placeholder="web-tools"
								class="mt-1"
							/>
						</div>
						<div>
							<Label for="server-name">Name</Label>
							<Input
								id="server-name"
								bind:value={newServer.name}
								placeholder="Web Tools"
								class="mt-1"
							/>
						</div>
						<div>
							<Label for="server-command">Command</Label>
							<Input
								id="server-command"
								bind:value={newServer.command}
								placeholder="python"
								class="mt-1"
							/>
						</div>
						<div>
							<Label for="server-args">Arguments (space-separated)</Label>
							<Input
								id="server-args"
								bind:value={newServer.args}
								placeholder="path/to/server.py"
								class="mt-1"
							/>
						</div>
						<div class="flex items-center gap-2">
							<Switch bind:checked={newServer.enabled} id="server-enabled" />
							<Label for="server-enabled">Enable immediately</Label>
						</div>
					</div>
					<div class="mt-4 flex gap-2">
						<Button onclick={handleAddServer}>Add Server</Button>
						<Button variant="outline" onclick={() => (isAdding = false)}>Cancel</Button>
					</div>
				</div>
			{/if}

			{#if mcpStore.servers.length > 0 && !isAdding}
				<Button variant="outline" class="w-full" onclick={() => (isAdding = true)}>
					<Plus class="mr-2 h-4 w-4" />
					Add Another Server
				</Button>
			{/if}

			<!-- Tool List -->
			{#if mcpStore.tools.length > 0}
				<div class="mt-6">
					<h3 class="mb-3 font-medium">Available Tools ({mcpStore.tools.length})</h3>
					<div class="space-y-2">
						{#each mcpStore.tools as tool}
							<div class="rounded-md border p-3">
								<div class="font-mono text-sm">{tool.function.name}</div>
								{#if tool.function.description}
									<p class="mt-1 text-xs text-muted-foreground">
										{tool.function.description}
									</p>
								{/if}
								<p class="mt-1 text-xs text-muted-foreground">Server: {tool.serverId}</p>
							</div>
						{/each}
					</div>
				</div>
			{/if}
		</div>
	</ScrollArea>

	<!-- Footer -->
	<div class="border-t p-4">
		<div class="flex justify-between">
			<div class="text-sm text-muted-foreground">
				{mcpStore.servers.filter((s) => s.enabled).length} enabled •
				{mcpStore.tools.length} tools available
			</div>
			<Button onclick={onSave}>Done</Button>
		</div>
	</div>
</div>
