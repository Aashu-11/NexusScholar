/* Hook wrapping chatStore for component consumption */
import { useEffect } from 'react';
import { useChatStore } from '../stores/chatStore';

export function useChat() {
    const store = useChatStore();

    useEffect(() => {
        store.loadConversations();
    }, []);

    return store;
}