<!--
	One build of Llama Chat, three ways to get it: bundled in llama-server (and so in the native apps), as
	`npx llama-chat`, and at chat.llama.app. The first comes with its own server; the other two are the client alone
	and connect to a server you add. Bundled ships today; the other two are planned.
-->
<script lang="ts">
	import { Badge } from '$lib/components/ui/badge';

	const channels = [
		{
			best: '“I run llama-server”, or a native app',
			name: 'In llama-server',
			served: 'llama-server, or the native apps, which run it',
			server: 'Comes with its own: it is the server',
			status: 'Today'
		},
		{
			best: 'A server elsewhere on your network, offline use',
			name: 'npx llama-chat',
			served: 'A small static server in the package',
			server: 'None: add a server by its address',
			status: 'Planned'
		},
		{
			best: 'Trying it with no install, phones',
			name: 'chat.llama.app',
			served: 'A static host',
			server: 'None: add a server by its address',
			status: 'Planned'
		}
	];
</script>

<div class="channels">
	{#each channels as c (c.name)}
		<section class="card card-surface rounded-card">
			<header>
				<h3><code>{c.name}</code></h3>

				<Badge variant={c.status === 'Today' ? 'default' : 'outline'}>{c.status}</Badge>
			</header>

			<dl>
				<dt>Served by</dt>

				<dd>{c.served}</dd>

				<dt>Server</dt>

				<dd>{c.server}</dd>

				<dt>Best for</dt>

				<dd>{c.best}</dd>
			</dl>
		</section>
	{/each}
</div>

<style>
	.channels {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr));
		gap: 8px;
	}

	.card {
		display: grid;
		gap: 12px;
		align-content: start;
		padding: 18px 20px;
	}

	header {
		display: flex;
		gap: 8px;
		justify-content: space-between;
		align-items: center;
	}

	h3 {
		margin: 0;
		font-size: 16px;
	}

	code {
		font: 600 15px var(--mono);
	}

	dl {
		display: grid;
		gap: 2px;
		margin: 0;
	}

	dt {
		margin-top: 8px;
		color: var(--figure-muted);
		font: 500 11px var(--mono);
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	dd {
		margin: 0;
		font-size: 15px;
		line-height: 1.45;
	}
</style>
