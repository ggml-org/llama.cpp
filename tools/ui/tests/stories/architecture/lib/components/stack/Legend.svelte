<!-- The colour key under the intro: library, apps, backends and docs -->
<div class="names">
	<span
		><i style:background="var(--lib)" class="dot"></i><b>Llama UI</b>&nbsp;UI and UX only, no
		business logic</span
	>

	<span
		><i style:background="var(--app)" class="dot"></i><b>Apps</b>&nbsp;own the business logic</span
	>

	<span><i style:background="var(--be)" class="dot"></i>Backends and docs</span>
</div>

<style>
	.names {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 18px;
		font-size: 14px;
	}
	span {
		display: inline-flex;
		align-items: center;
		gap: 8px;
	}
	.dot {
		width: 10px;
		height: 10px;
		border-radius: 3px;
		display: inline-block;
	}
</style>
