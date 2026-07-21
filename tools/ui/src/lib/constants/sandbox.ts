import { BuiltInTool, JsonSchemaType, ToolCallType } from '$lib/enums';
import type { OpenAIToolDefinition } from '$lib/types';

export const SANDBOX_TOOL_NAME = BuiltInTool.RUN_JAVASCRIPT;

export const SANDBOX_TIMEOUT_MS_DEFAULT = 10000;

export const SANDBOX_TIMEOUT_MS_MAX = 30000;

export const SANDBOX_OUTPUT_MAX_CHARS = 8192;

export const SANDBOX_EMPTY_OUTPUT = '(no output)';

export const SANDBOX_TRUNCATION_NOTICE = '[output truncated]';

export const SANDBOX_TOOL_DEFINITION: OpenAIToolDefinition = {
	type: ToolCallType.FUNCTION,
	function: {
		name: SANDBOX_TOOL_NAME,
		description: `Execute JS in a sandboxed browser worker (no DOM/page access). Top-level await ok; console.log for intermediates; top-level return is captured as result.
Symbolic/numeric math via \`nerdamer\` (pre-loaded, do not require, use it directly). Object.keys(nerdamer).filter(k=>typeof nerdamer[k]==='function') list functions.
nerdamer('diff(sin(x)/x,x)') or nerdamer.diff('sin(x)/x','x') → Expression; convert with .toString()/.text()/.toTeX(), or .evaluate() (→ still Expression, then .toString()).
nerdamer(expr,{x:2}) substitutes only; chain .evaluate() or pass 'numer' for numeric result.
solve(expr,var)→Symbol[]; solveEquations([eq1,..])→[[var,val],..] pairs.
Functions: simplify/expand/factor(expr), diff(expr,var[,n]), integrate(expr,var), defint(expr,from,to,var), limit(expr,var,to), laplace(expr,t,s), ilt(expr,s,t), gcd/lcm(a,b), roots/coeffs/partfrac(expr,var), pfactor(n), numer/decimals/erf(expr), product/sum(expr,var,from,to), mean/median/stdev/variance(...vals)`,
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
