import { useState } from 'react';
import { MessageExtra } from '../utils/types';
import toast from 'react-hot-toast';
import { useAppContext } from '../utils/app.context';

// Interface describing the API returned by the hook
export interface ChatExtraContextApi {
  items?: MessageExtra[]; // undefined if empty, similar to Message['extra']
  addItems: (items: MessageExtra[]) => void;
  removeItem: (idx: number) => void;
  clearItems: () => void;
  onFileAdded: (files: File[]) => void; // used by "upload" button
}

export function useChatExtraContext(): ChatExtraContextApi {
  const { serverProps } = useAppContext();
  const [items, setItems] = useState<MessageExtra[]>([]);

  const addItems = (newItems: MessageExtra[]) => {
    setItems((prev) => [...prev, ...newItems]);
  };

  const removeItem = (idx: number) => {
    setItems((prev) => prev.filter((_, i) => i !== idx));
  };

  const clearItems = () => {
    setItems([]);
  };

  const onFileAdded = (files: File[]) => {
    for (const file of files) {
      const mimeType = file.type;
      console.debug({ mimeType, file });
      if (file.size > 10 * 1024 * 1024) {
        toast.error('File is too large. Maximum size is 10MB.');
        break;
      }

      if (mimeType.startsWith('image/')) {
        if (!serverProps?.modalities?.vision) {
          toast.error('Multimodal is not supported by this server or model.');
          break;
        }
        const reader = new FileReader();
        reader.onload = async (event) => {
          if (event.target?.result) {
            let base64Url = event.target.result as string;

            if (mimeType === 'image/svg+xml') {
              // Convert SVG to PNG
              base64Url = await svgBase64UrlToPngDataURL(base64Url);
            }

            addItems([
              {
                type: 'imageFile',
                name: file.name,
                base64Url,
              },
            ]);
          }
        };
        reader.readAsDataURL(file);
      } else if (
        mimeType.startsWith('video/') ||
        mimeType.startsWith('audio/')
      ) {
        toast.error('Video and audio files are not supported yet.');
        break;
      } else if (mimeType.startsWith('application/pdf')) {
        toast.error('PDF files are not supported yet.');
        break;
      } else {
        // Because there can be many text file types (like code file), we will not check the mime type
        // and will just check if the file is not binary.
        const reader = new FileReader();
        reader.onload = (event) => {
          if (event.target?.result) {
            const content = event.target.result as string;
            if (!isLikelyNotBinary(content)) {
              toast.error('File is binary. Please upload a text file.');
              return;
            }
            addItems([
              {
                type: 'textFile',
                name: file.name,
                content,
              },
            ]);
          }
        };
        reader.readAsText(file);
      }
    }
  };

  return {
    items: items.length > 0 ? items : undefined,
    addItems,
    removeItem,
    clearItems,
    onFileAdded,
  };
}

// WARN: vibe code below
// This code is a heuristic to determine if a string is likely not binary.
// It is necessary because input file can have various mime types which we don't have time to investigate.
// For example, a python file can be text/plain, application/x-python, etc.
export function isLikelyNotBinary(str: string): boolean {
  const options = {
    prefixLength: 1024 * 10, // Check the first 10KB of the string
    suspiciousCharThresholdRatio: 0.15, // Allow up to 15% suspicious chars
    maxAbsoluteNullBytes: 2,
  };

  if (!str) {
    return true; // Empty string is considered "not binary" or trivially text.
  }

  const sampleLength = Math.min(str.length, options.prefixLength);
  if (sampleLength === 0) {
    return true; // Effectively an empty string after considering prefixLength.
  }

  let suspiciousCharCount = 0;
  let nullByteCount = 0;

  for (let i = 0; i < sampleLength; i++) {
    const charCode = str.charCodeAt(i);

    // 1. Check for Unicode Replacement Character (U+FFFD)
    // This is a strong indicator if the string was created from decoding bytes as UTF-8.
    if (charCode === 0xfffd) {
      suspiciousCharCount++;
      continue;
    }

    // 2. Check for Null Bytes (U+0000)
    if (charCode === 0x0000) {
      nullByteCount++;
      // We also count nulls towards the general suspicious character count,
      // as they are less common in typical text files.
      suspiciousCharCount++;
      continue;
    }

    // 3. Check for C0 Control Characters (U+0001 to U+001F)
    // Exclude common text control characters: TAB (9), LF (10), CR (13).
    // We can also be a bit lenient with BEL (7) and BS (8) which sometimes appear in logs.
    if (charCode < 32) {
      if (
        charCode !== 9 && // TAB
        charCode !== 10 && // LF
        charCode !== 13 && // CR
        charCode !== 7 && // BEL (Bell) - sometimes in logs
        charCode !== 8 // BS (Backspace) - less common, but possible
      ) {
        suspiciousCharCount++;
      }
    }
    // Characters from 32 (space) up to 126 (~) are printable ASCII.
    // Characters 127 (DEL) is a control character.
    // Characters >= 128 are extended ASCII / multi-byte Unicode.
    // If they resulted in U+FFFD, we caught it. Otherwise, they are valid
    // (though perhaps unusual) Unicode characters from JS's perspective.
    // The main concern is if those higher characters came from misinterpreting
    // a single-byte encoding as UTF-8, which again, U+FFFD would usually flag.
  }

  // Check absolute null byte count
  if (nullByteCount > options.maxAbsoluteNullBytes) {
    return false; // Too many null bytes is a strong binary indicator
  }

  // Check ratio of suspicious characters
  const ratio = suspiciousCharCount / sampleLength;
  return ratio <= options.suspiciousCharThresholdRatio;
}

// WARN: vibe code below
// Converts a Base64URL encoded SVG string to a PNG Data URL using browser Canvas API.
function svgBase64UrlToPngDataURL(base64UrlSvg: string): Promise<string> {
  const backgroundColor = 'white'; // Default background color for PNG

  return new Promise((resolve, reject) => {
    try {
      const img = new Image();

      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        if (!ctx) {
          reject(new Error('Failed to get 2D canvas context.'));
          return;
        }

        // Use provided dimensions or SVG's natural dimensions, with fallbacks
        // Fallbacks (e.g., 300x300) are for SVGs without explicit width/height
        // or when naturalWidth/Height might be 0 before full processing.
        const targetWidth = img.naturalWidth || 300;
        const targetHeight = img.naturalHeight || 300;

        canvas.width = targetWidth;
        canvas.height = targetHeight;

        if (backgroundColor) {
          ctx.fillStyle = backgroundColor;
          ctx.fillRect(0, 0, canvas.width, canvas.height);
        }

        ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
        resolve(canvas.toDataURL('image/png'));
      };

      img.onerror = () => {
        reject(
          new Error('Failed to load SVG image. Ensure the SVG data is valid.')
        );
      };

      // Load SVG string into an Image element
      img.src = base64UrlSvg;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const errorMessage = `Error converting SVG to PNG: ${message}`;
      toast.error(errorMessage);
      reject(new Error(errorMessage));
    }
  });
}
