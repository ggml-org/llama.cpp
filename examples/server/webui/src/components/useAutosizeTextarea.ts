import { useEffect, useRef, useState, useCallback } from 'react';

const adjustTextareaHeight = (textarea: HTMLTextAreaElement | null) => {
  if (!textarea) {
    return;
  }

  const computedStyle = window.getComputedStyle(textarea);
  const currentMaxHeight = computedStyle.maxHeight;

  textarea.style.maxHeight = 'none';
  textarea.style.height = 'auto';  

  const scrollH = textarea.scrollHeight;  

  textarea.style.height = `${scrollH}px`;  
  textarea.style.maxHeight = currentMaxHeight;  
};

export interface AutosizeTextareaApi {
  value: () => string;
  setValue: (value: string) => void;
  focus: () => void;
  ref: React.RefObject<HTMLTextAreaElement>;
  onInput: (event: React.FormEvent<HTMLTextAreaElement>) => void;  
}

export function useAutosizeTextarea(initValue: string): AutosizeTextareaApi {
  const [savedInitValue, setSavedInitValue] = useState<string>(initValue);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      if (typeof savedInitValue === 'string' && savedInitValue.length > 0) {
        textarea.value = savedInitValue;
        setTimeout(() => adjustTextareaHeight(textarea), 0);
        setSavedInitValue('');
      } else {
         setTimeout(() => adjustTextareaHeight(textarea), 0);
      }
    } 
  }, [textareaRef, savedInitValue]);

  const handleInput = useCallback((event: React.FormEvent<HTMLTextAreaElement>) => {
    adjustTextareaHeight(event.currentTarget);
  }, []);  

  return {
    value: () => {
      return textareaRef.current?.value ?? '';
    },
    setValue: (value: string) => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.value = value;
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