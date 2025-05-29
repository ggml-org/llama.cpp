import { ToolCallRequest, ToolCallOutput, ToolCallParameters } from '../types';
import { AgentTool } from './agent_tool';

// Import the HTML content as a raw string
import iframeHTMLContent from '../../assets/iframe_sandbox.html?raw';
import StorageUtils from '../storage';

interface IframeMessage {
  call_id: string;
  output?: string;
  error?: string;
  command?: 'executeCode' | 'iframeReady';
  code?: string;
}

export class JSReplAgentTool extends AgentTool {
  private static readonly ID = 'javascript_interpreter';
  private iframe: HTMLIFrameElement | null = null;
  private iframeReadyPromise: Promise<void> | null = null;
  private resolveIframeReady: (() => void) | null = null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private rejectIframeReady: ((reason?: any) => void) | null = null;
  private pendingCalls = new Map<string, (output: ToolCallOutput) => void>();
  private messageHandler:
    | ((event: MessageEvent<IframeMessage>) => void)
    | null = null;

  constructor() {
    super(
      JSReplAgentTool.ID,
      'Javascript interpreter',
      'Executes JavaScript code in a sandboxed iframe. The code should be self-contained valid javascript. Only console.log(variable) and final result are included in response content.',
      {
        type: 'object',
        properties: {
          code: {
            type: 'string',
            description: 'Valid JavaScript code to execute.',
          },
        },
        required: ['code'],
      } as ToolCallParameters,
      () => StorageUtils.getConfig().toolJsReplEnabled
    );
    this.initIframe();
  }

  private initIframe(): void {
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      console.warn(
        'JSReplAgentTool: Not in a browser environment, iframe will not be created.'
      );
      return;
    }

    this.iframeReadyPromise = new Promise<void>((resolve, reject) => {
      this.resolveIframeReady = resolve;
      this.rejectIframeReady = reject;
    });

    this.messageHandler = (event: MessageEvent<IframeMessage>) => {
      if (
        !event.data ||
        !this.iframe ||
        !this.iframe.contentWindow ||
        event.source !== this.iframe.contentWindow
      ) {
        return;
      }

      const { command, call_id, output, error } = event.data;
      if (command === 'iframeReady' && call_id === 'initial_ready') {
        if (this.resolveIframeReady) {
          this.resolveIframeReady();
          this.resolveIframeReady = null;
          this.rejectIframeReady = null;
        }
        return;
      }
      if (typeof call_id !== 'string') {
        return;
      }
      if (this.pendingCalls.has(call_id)) {
        const callback = this.pendingCalls.get(call_id)!;
        callback({
          type: 'function_call_output',
          call_id: call_id,
          output: error ? `Error: ${error}` : (output ?? ''),
        } as ToolCallOutput);
        this.pendingCalls.delete(call_id);
      }
    };
    window.addEventListener('message', this.messageHandler);

    this.iframe = document.createElement('iframe');
    this.iframe.style.display = 'none';
    this.iframe.sandbox.add('allow-scripts');

    // Use srcdoc with the imported HTML content
    this.iframe.srcdoc = iframeHTMLContent;

    document.body.appendChild(this.iframe);

    setTimeout(() => {
      if (this.rejectIframeReady) {
        this.rejectIframeReady(new Error('Iframe readiness timeout'));
        this.resolveIframeReady = null;
        this.rejectIframeReady = null;
      }
    }, 5000);
  }

  async _process(tc: ToolCallRequest): Promise<ToolCallOutput> {
    let error = null;
    if (
      typeof window === 'undefined' ||
      !this.iframe ||
      !this.iframe.contentWindow ||
      !this.iframeReadyPromise
    ) {
      error =
        'Error: JavaScript interpreter is not available or iframe not ready.';
    }

    try {
      await this.iframeReadyPromise;
    } catch (e) {
      error = `Error: Iframe for JavaScript interpreter failed to initialize. ${(e as Error).message}`;
    }

    let args;
    try {
      args = JSON.parse(tc.function.arguments);
    } catch (e) {
      error = `Error: Could not parse arguments for tool call. ${(e as Error).message}`;
    }

    const codeToExecute = args.code;
    if (typeof codeToExecute !== 'string') {
      error = 'Error: "code" argument must be a string.';
    }

    if (error) {
      return {
        type: 'function_call_output',
        call_id: tc.call_id,
        output: error,
      } as ToolCallOutput;
    }

    return new Promise<ToolCallOutput>((resolve) => {
      this.pendingCalls.set(tc.call_id, resolve);
      const message: IframeMessage = {
        command: 'executeCode',
        code: codeToExecute,
        call_id: tc.call_id,
      };
      this.iframe!.contentWindow!.postMessage(message, '*');
    });
  }
}
