import { BuiltInTool, JsonSchemaType, ToolCallType } from '$lib/enums';
import type { OpenAIToolDefinition } from '$lib/types';

export const SANDBOX_TOOL_NAME = BuiltInTool.RUN_JAVASCRIPT;

/** Name of the browser-only set_working_directory frontend tool. */
export const WORKING_DIRECTORY_TOOL_NAME = BuiltInTool.SET_WORKING_DIRECTORY;

export const SANDBOX_TIMEOUT_MS_DEFAULT = 10000;

export const SANDBOX_TIMEOUT_MS_MAX = 30000;

export const SANDBOX_OUTPUT_MAX_CHARS = 8192;

export const SANDBOX_EMPTY_OUTPUT = '(no output)';

export const SANDBOX_TRUNCATION_NOTICE = '[output truncated]';

const NERDAMER_DESCRIPTION = `
Symbolic/numeric math via \`nerdamer\`
nerdamer(expr,subs?,opts?)/nerdamer.func(...)→Expression Format via .text(fmt?) (fmt: 'decimals'|'fractions'|'scientific') eval via .evaluate(subs?)
nerdamer(expr,{x:2}) substitutes numeric via opts 'numer' or .evaluate()
simplify/expand/factor(expr) div/gcd/lcm(...) coeffs/partfrac(expr,var)
diff/integrate(expr,var) defint(expr,lo,hi,var?) sum/product(expr,var,lo,hi) limit(expr,var,pt)
solve(expr,var) solveEquations([eq1,eq2],[var1,var2])
polarform/rectform/arg/realpart/imagpart(z)
set/get Var/Constant(name,val?) setFunction(name,[params],body)
IMPORTANT:Identifier 'nerdamer' has already been declared, use it directly`;

/**
 * Build the sandbox tool definition. When `includeSymbolicMath` is true,
 * the description includes nerdamer API documentation; otherwise it
 * describes a plain JavaScript sandbox.
 */
export function buildSandboxToolDefinition(includeSymbolicMath: boolean): OpenAIToolDefinition {
	return {
		type: ToolCallType.FUNCTION,
		function: {
			name: SANDBOX_TOOL_NAME,
			description: includeSymbolicMath
				? `Execute JS in a sandboxed browser worker (no DOM/page access). Top-level await ok; console.log for intermediates; top-level return is captured as result.${NERDAMER_DESCRIPTION}`
				: 'Execute JS in a sandboxed browser worker (no DOM/page access). Top-level await ok; console.log for intermediates; top-level return is captured as result.',
			parameters: {
				type: JsonSchemaType.OBJECT,
				properties: {
					code: {
						type: JsonSchemaType.STRING,
						description: 'JavaScript source to execute'
					},
					timeout_ms: {
						type: JsonSchemaType.NUMBER,
						description: `Execution timeout in milliseconds, default ${SANDBOX_TIMEOUT_MS_DEFAULT}, max ${SANDBOX_TIMEOUT_MS_MAX}`
					}
				},
				required: ['code']
			}
		}
	};
}

/** @deprecated Use {@link buildSandboxToolDefinition} instead. Kept for backward compatibility. */
export const SANDBOX_TOOL_DEFINITION = buildSandboxToolDefinition(true);

/**
 * Build the OpenAI-compatible tool definition for the browser-only
 * `set_working_directory` frontend tool. This tool writes the supplied
 * path into the active conversation's IndexedDB row (via
 * `conversationsStore.setWorkingDirectory`) so that subsequent server-side
 * file operations use it as their base directory.
 */
export function buildSetWorkingDirectoryToolDefinition(): OpenAIToolDefinition {
	return {
		type: ToolCallType.FUNCTION,
		function: {
			name: WORKING_DIRECTORY_TOOL_NAME,
			description:
				'Set the working directory for the current conversation. The directory is persisted in the conversation state (IndexedDB) and used as the base path for subsequent file system tool calls. Pass an empty string to clear the working directory.',
			parameters: {
				type: JsonSchemaType.OBJECT,
				properties: {
					path: {
						type: JsonSchemaType.STRING,
						description: 'Absolute path to set as the working directory. Pass an empty string to clear it.'
					}
				},
				required: ['path']
			}
		}
	};
}
