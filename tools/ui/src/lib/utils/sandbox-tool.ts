import {
	SANDBOX_TIMEOUT_MS_DEFAULT,
	SANDBOX_TIMEOUT_MS_MAX,
	SANDBOX_TOOL_NAME
} from '$lib/constants';
import { JsonSchemaType, ToolCallType } from '$lib/enums';
import type { OpenAIToolDefinition } from '$lib/types';

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
		function: {
			description: includeSymbolicMath
				? `Execute JS in a sandboxed browser worker (no DOM/page access). Top-level await ok; console.log for intermediates; top-level return is captured as result.${NERDAMER_DESCRIPTION}`
				: 'Execute JS in a sandboxed browser worker (no DOM/page access). Top-level await ok; console.log for intermediates; top-level return is captured as result.',
			name: SANDBOX_TOOL_NAME,
			parameters: {
				properties: {
					code: {
						description: 'JavaScript source to execute',
						type: JsonSchemaType.STRING
					},
					timeout_ms: {
						description: `Execution timeout in milliseconds, default ${SANDBOX_TIMEOUT_MS_DEFAULT}, max ${SANDBOX_TIMEOUT_MS_MAX}`,
						type: JsonSchemaType.NUMBER
					}
				},
				required: ['code'],
				type: JsonSchemaType.OBJECT
			}
		},
		type: ToolCallType.FUNCTION
	};
}

/** @deprecated Use {@link buildSandboxToolDefinition} instead. Kept for backward compatibility. */
export const SANDBOX_TOOL_DEFINITION = buildSandboxToolDefinition(true);
