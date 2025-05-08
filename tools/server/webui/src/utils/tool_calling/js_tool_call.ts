import StorageUtils from '../storage';
import { ToolCall, ToolCallOutput, ToolCallSpec } from '../types';
import { AgentTool } from './available_tools';

class JSReplAgentTool extends AgentTool {
  private static readonly id = 'javascript_interpreter';
  private fakeLogger: FakeConsoleLog;

  constructor() {
    super(
      JSReplAgentTool.id,
      () => StorageUtils.getConfig().jsInterpreterToolUse
    );
    this.fakeLogger = new FakeConsoleLog();
  }

  _process(tc: ToolCall): ToolCallOutput {
    const args = JSON.parse(tc.function.arguments);
    console.log('Arguments for tool call:');
    console.log(args);

    // Redirect console.log which agent will use to
    // the fake logger so that later we can get the content
    const originalConsoleLog = console.log;
    console.log = this.fakeLogger.log;

    let result = '';
    try {
      // Evaluate the provided agent code
      result = eval(args.code);
    } catch (err) {
      result = String(err);
    } finally {
      // Ensure original console.log is restored even if eval throws
      console.log = originalConsoleLog;
    }

    result = this.fakeLogger.content + result;

    this.fakeLogger.clear();

    return { call_id: tc.call_id, output: result } as ToolCallOutput;
  }

  public specs(): ToolCallSpec {
    return {
      type: 'function',
      function: {
        name: this.id,
        description:
          'Executes JavaScript code in the browser console. The code should be self-contained valid javascript. You can use console.log(variable) to print out intermediate values..',
        parameters: {
          type: 'object',
          properties: {
            code: {
              type: 'string',
              description: 'Valid JavaScript code to execute.',
            },
          },
          required: ['code'],
        },
      },
    };
  }
}
export const jsAgentTool = new JSReplAgentTool();

class FakeConsoleLog {
  private _content: string = '';

  public get content(): string {
    return this._content;
  }

  // Use an arrow function for log to correctly bind 'this'
  public log = (...args: any[]): void => {
    // Convert arguments to strings and join them.
    this._content += args.map((arg) => String(arg)).join(' ') + '\n';
  };

  public clear = (): void => {
    this._content = '';
  };
}
