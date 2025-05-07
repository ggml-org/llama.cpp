export const JS_TOOL_CALL_SPEC = {
  type: 'function',
  function: {
    name: 'javascript_interpreter',
    description:
      'Executes JavaScript code in the browser console and returns the output or error. The code should be self-contained. Use JSON.stringify for complex return objects.',
    parameters: {
      type: 'object',
      properties: {
        code: {
          type: 'string',
          description: 'The JavaScript code to execute.',
        },
      },
      required: ['code'],
    },
  },
};
