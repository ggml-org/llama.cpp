import { useEffect, useRef, useState, useCallback } from 'react';

// Media Query for detecting "large" screens (matching Tailwind's lg: breakpoint)
const LARGE_SCREEN_MQ = '(min-width: 1024px)';

// Calculates and sets the textarea height based on its scrollHeight
const adjustTextareaHeight = (textarea: HTMLTextAreaElement | null) => {
  if (!textarea) return;

  // Only perform auto-sizing on large screens
  if (!window.matchMedia(LARGE_SCREEN_MQ).matches) {
    // On small screens, reset inline height and max-height styles.
    // This allows CSS (e.g., `rows` attribute or classes) to control the height,
    // and enables manual resizing if `resize-vertical` is set.
    textarea.style.height = ''; // Use 'auto' or '' to reset
    textarea.style.maxHeight = '';
    return; // Do not adjust height programmatically on small screens
  }

  const computedStyle = window.getComputedStyle(textarea);
  // Get the max-height specified by CSS (e.g., from `lg:max-h-48`)
  const currentMaxHeight = computedStyle.maxHeight;

  // Temporarily remove max-height to allow scrollHeight to be calculated correctly
  textarea.style.maxHeight = 'none';
  // Reset height to 'auto' to measure the actual scrollHeight needed
  textarea.style.height = 'auto';
  // Set the height to the calculated scrollHeight
  textarea.style.height = `${textarea.scrollHeight}px`;
  // Re-apply the original max-height from CSS to enforce the limit
  textarea.style.maxHeight = currentMaxHeight;
};

// Interface describing the API returned by the hook
export interface ChatTextareaApi {
  value: () => string;
  setValue: (value: string) => void;
  focus: () => void;
  ref: React.RefObject<HTMLTextAreaElement>;
  onInput: (event: React.FormEvent<HTMLTextAreaElement>) => void; // Input handler
}

// This is a workaround to prevent the textarea from re-rendering when the inner content changes
// See https://github.com/ggml-org/llama.cpp/pull/12299
// combined now with auto-sizing logic.
export function useChatTextarea(initValue: string): ChatTextareaApi {
  const [savedInitValue, setSavedInitValue] = useState<string>(initValue);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Effect to set initial value and height on mount or when initValue changes
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      if (typeof savedInitValue === 'string' && savedInitValue.length > 0) {
        textarea.value = savedInitValue;
        // Call adjustTextareaHeight - it will check screen size internally
        setTimeout(() => adjustTextareaHeight(textarea), 0);
        setSavedInitValue(''); // Reset after applying
      } else {
        // Adjust height even if there's no initial value (for initial render)
        setTimeout(() => adjustTextareaHeight(textarea), 0);
      }
    }
  }, [textareaRef, savedInitValue]); // Depend on ref and savedInitValue

  const handleInput = useCallback(
    (event: React.FormEvent<HTMLTextAreaElement>) => {
      // Call adjustTextareaHeight on every input - it will decide whether to act
      adjustTextareaHeight(event.currentTarget);
    },
    []
  );

  return {
    // Method to get the current value directly from the textarea
    value: () => {
      return textareaRef.current?.value ?? '';
    },
    // Method to programmatically set the value and trigger height adjustment
    setValue: (value: string) => {
      const textarea = textareaRef.current;
      if (textarea) {
        textarea.value = value;
        // Call adjustTextareaHeight - it will check screen size internally
        setTimeout(() => adjustTextareaHeight(textarea), 0);
      }
    },
    focus: () => {
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    },
    ref: textareaRef,
    onInput: handleInput,
  };
}
