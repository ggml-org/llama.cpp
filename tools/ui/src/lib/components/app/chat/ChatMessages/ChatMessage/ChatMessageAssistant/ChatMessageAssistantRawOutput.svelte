<script lang="ts">
	import { deriveAgenticSections } from '$lib/utils';
	import { AgenticSectionType } from '$lib/enums';
	import { REASONING_TAGS } from '$lib/constants/agentic';

	interface Props {
		message: DatabaseMessage;
		toolMessages?: DatabaseMessage[];
	}

	let { message, toolMessages = [] }: Props = $props();

	let rawOutputContent = $derived.by(() => {
		const sections = deriveAgenticSections(message, toolMessages, [], false);
		const parts: string[] = [];

		for (const section of sections) {
			switch (section.type) {
				case AgenticSectionType.REASONING:
				case AgenticSectionType.REASONING_PENDING:
					parts.push(`${REASONING_TAGS.START}\n${section.content}\n${REASONING_TAGS.END}`);
					break;

				case AgenticSectionType.TEXT:
					parts.push(section.content);
					break;

				case AgenticSectionType.TOOL_CALL:
				case AgenticSectionType.TOOL_CALL_PENDING:
				case AgenticSectionType.TOOL_CALL_STREAMING: {
					const callObj: Record<string, unknown> = { name: section.toolName };

					if (section.toolArgs) {
						try {
							callObj.arguments = JSON.parse(section.toolArgs);
						} catch {
							callObj.arguments = section.toolArgs;
						}
					}

					parts.push(JSON.stringify(callObj, null, 2));

					if (section.toolResult) {
						parts.push(`[Tool Result]\n${section.toolResult}`);
					}

					break;
				}
			}
		}

		return parts.join('\n\n\n');
	});
</script>

<pre class="raw-output">{rawOutputContent || ''}</pre>

<style>
	.raw-output {
		width: 100%;
		max-width: 48rem;
		margin-top: 1.5rem;
		padding: 1rem 1.25rem;
		border-radius: 1rem;
		background: hsl(var(--muted) / 0.3);
		color: var(--foreground);
		font-size: 0.875rem;
		line-height: 1.6;
		white-space: pre-wrap;
		word-break: break-word;
	}
</style>
