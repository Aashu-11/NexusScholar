/* Auto-scroll and streaming state helpers */
import { useEffect, useRef } from 'react';
import { useChatStore } from '../stores/chatStore';

export function useAutoScroll() {
    const ref = useRef<HTMLDivElement>(null);
    const messages = useChatStore(s => s.messages);

    useEffect(() => {
        ref.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    return ref;
}

export function useStreamingState() {
    return {
        isStreaming: useChatStore(s => s.isStreaming),
        pipelineStatus: useChatStore(s => s.pipelineStatus),
        currentIntent: useChatStore(s => s.currentIntent),
    };
}