import StorageUtils from '../storage';
import { ToolCall, ToolCallOutput, ToolCallParameters } from '../types';
import { AgentTool } from './agent_tool';

export class JSReplAgentTool extends AgentTool {
  private static readonly id = 'javascript_interpreter';
  private fakeLogger: FakeConsoleLog;

  constructor() {
    super(
      JSReplAgentTool.id,
      () => StorageUtils.getConfig().jsInterpreterToolUse,
      'Executes JavaScript code in the browser console. The code should be self-contained valid javascript. You can use console.log(variable) to print out intermediate values..',
      {
        type: 'object',
        properties: {
          code: {
            type: 'string',
            description: 'Valid JavaScript code to execute.',
          },
        },
        required: ['code'],
      } as ToolCallParameters
    );
    this.fakeLogger = new FakeConsoleLog();
  }

  _process(tc: ToolCall): ToolCallOutput {
    const args = JSON.parse(tc.function.arguments);

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
    }

    console.log = originalConsoleLog;
    result = this.fakeLogger.content + result;

    this.fakeLogger.clear();

    return { call_id: tc.call_id, output: result } as ToolCallOutput;
  }
}

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
