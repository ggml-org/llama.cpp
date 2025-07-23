// @ts-expect-error this package does not have typing
import TextLineStream from 'textlinestream';
import {
  APIMessage,
  APIMessageContentPart,
  LlamaCppServerProps,
  Message,
} from './types';

// ponyfill for missing ReadableStream asyncIterator on Safari
import { asyncIterator } from '@sec-ant/readable-stream/ponyfill/asyncIterator';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const isString = (x: any) => !!x.toLowerCase;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const isBoolean = (x: any) => x === true || x === false;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const isNumeric = (n: any) => !isString(n) && !isNaN(n) && !isBoolean(n);
export const escapeAttr = (str: string) =>
  str.replace(/>/g, '&gt;').replace(/"/g, '&quot;');

// wrapper for SSE
export async function* getSSEStreamAsync(fetchResponse: Response) {
  if (!fetchResponse.body) throw new Error('Response body is empty');
  const lines: ReadableStream<string> = fetchResponse.body
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(new TextLineStream());
  // @ts-expect-error asyncIterator complains about type, but it should work
  for await (const line of asyncIterator(lines)) {
    //if (isDev) console.log({ line });
    if (line.startsWith('data:') && !line.endsWith('[DONE]')) {
      const data = JSON.parse(line.slice(5));
      yield data;
    } else if (line.startsWith('error:')) {
      const data = JSON.parse(line.slice(6));
      throw new Error(data.message || 'Unknown error');
    }
  }
}

// copy text to clipboard
export const copyStr = (textToCopy: string) => {
  // Navigator clipboard api needs a secure context (https)
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(textToCopy);
  } else {
    // Use the 'out of viewport hidden text area' trick
    const textArea = document.createElement('textarea');
    textArea.value = textToCopy;
    // Move textarea out of the viewport so it's not visible
    textArea.style.position = 'absolute';
    textArea.style.left = '-999999px';
    document.body.prepend(textArea);
    textArea.select();
    document.execCommand('copy');
  }
};

/**
 * filter out redundant fields upon sending to API
 * also format extra into text
 */
export function normalizeMsgsForAPI(messages: Readonly<Message[]>) {
  return messages.map((msg) => {
    if (msg.role !== 'user' || !msg.extra) {
      return {
        role: msg.role,
        content: msg.content,
      } as APIMessage;
    }

    // extra content first, then user text message in the end
    // this allow re-using the same cache prefix for long context
    const contentArr: APIMessageContentPart[] = [];

    for (const extra of msg.extra ?? []) {
      if (extra.type === 'context') {
        contentArr.push({
          type: 'text',
          text: extra.content,
        });
      } else if (extra.type === 'textFile') {
        contentArr.push({
          type: 'text',
          text: `File: ${extra.name}\nContent:\n\n${extra.content}`,
        });
      } else if (extra.type === 'imageFile') {
        contentArr.push({
          type: 'image_url',
          image_url: { url: extra.base64Url },
        });
      } else if (extra.type === 'audioFile') {
        contentArr.push({
          type: 'input_audio',
          input_audio: {
            data: extra.base64Data,
            format: /wav/.test(extra.mimeType) ? 'wav' : 'mp3',
          },
        });
      } else {
        throw new Error('Unknown extra type');
      }
    }

    // add user message to the end
    contentArr.push({
      type: 'text',
      text: msg.content,
    });

    return {
      role: msg.role,
      content: contentArr,
    };
  }) as APIMessage[];
}

/**
 * recommended for DeepsSeek-R1, filter out content between <think> and </think> tags
 */
export function filterThoughtFromMsgs(messages: APIMessage[]) {
  console.debug({ messages });
  return messages.map((msg) => {
    if (msg.role !== 'assistant') {
      return msg;
    }
    // assistant message is always a string
    const contentStr = msg.content as string;
    return {
      role: msg.role,
      content:
        msg.role === 'assistant'
          ? contentStr.split('</think>').at(-1)!.trim()
          : contentStr,
    } as APIMessage;
  });
}

export function classNames(classes: Record<string, boolean>): string {
  return Object.entries(classes)
    .filter(([_, value]) => value)
    .map(([key, _]) => key)
    .join(' ');
}

export const delay = (ms: number) =>
  new Promise((resolve) => setTimeout(resolve, ms));

export const throttle = <T extends unknown[]>(
  callback: (...args: T) => void,
  delay: number
) => {
  let isWaiting = false;

  return (...args: T) => {
    if (isWaiting) {
      return;
    }

    callback(...args);
    isWaiting = true;

    setTimeout(() => {
      isWaiting = false;
    }, delay);
  };
};

export const cleanCurrentUrl = (removeQueryParams: string[]) => {
  const url = new URL(window.location.href);
  removeQueryParams.forEach((param) => {
    url.searchParams.delete(param);
  });
  window.history.replaceState({}, '', url.toString());
};

export const getServerProps = async (
  baseUrl: string,
  apiKey?: string
): Promise<LlamaCppServerProps> => {
  try {
    const response = await fetch(`${baseUrl}/props`, {
      headers: {
        'Content-Type': 'application/json',
        ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
      },
    });
    if (!response.ok) {
      throw new Error('Failed to fetch server props');
    }
    const data = await response.json();
    return data as LlamaCppServerProps;
  } catch (error) {
    console.error('Error fetching server props:', error);
    throw error;
  }
};
