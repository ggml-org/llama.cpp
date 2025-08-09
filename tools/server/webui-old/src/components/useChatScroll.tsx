import React, { useEffect } from 'react';
import { throttle } from '../utils/misc';

export const scrollToBottom = (requiresNearBottom: boolean, delay?: number) => {
  const mainScrollElem = document.getElementById('main-scroll');
  if (!mainScrollElem) return;
  const spaceToBottom =
    mainScrollElem.scrollHeight -
    mainScrollElem.scrollTop -
    mainScrollElem.clientHeight;
  if (!requiresNearBottom || spaceToBottom < 100) {
    setTimeout(
      () => mainScrollElem.scrollTo({ top: mainScrollElem.scrollHeight }),
      delay ?? 80
    );
  }
};

const scrollToBottomThrottled = throttle(scrollToBottom, 80);

export function useChatScroll(msgListRef: React.RefObject<HTMLDivElement>) {
  useEffect(() => {
    if (!msgListRef.current) return;

    const resizeObserver = new ResizeObserver((_) => {
      scrollToBottomThrottled(true, 10);
    });

    resizeObserver.observe(msgListRef.current);
    return () => {
      resizeObserver.disconnect();
    };
  }, [msgListRef]);
}
