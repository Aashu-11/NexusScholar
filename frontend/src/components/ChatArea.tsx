import React from 'react';
import type { Message } from '../types';
import MessageBubble from './MessageBubble';
import WelcomeScreen from './WelcomeScreen';
import { useAutoScroll } from '../hooks/useStreaming';

interface Props {
    messages: Message[];
    pipelineStatus: string | null;
    onSend: (query: string, recencyFilter?: string) => void;
    onCitationClick: (num: string) => void;
}

export default function ChatArea({ messages, pipelineStatus, onSend, onCitationClick }: Props) {
    const scrollRef = useAutoScroll();

    if (messages.length === 0) {
        return (
            <div className="chat-messages">
                <WelcomeScreen onSend={onSend} />
            </div>
        );
    }

    return (
        <div className="chat-messages">
            <div className="messages-container">
                {messages.map(msg => (
                    <MessageBubble key={msg.id} message={msg} onCitationClick={onCitationClick} />
                ))}

                {pipelineStatus && (
                    <div className="pipeline-status">
                        <div className="status-dot" />
                        {pipelineStatus}
                    </div>
                )}

                <div ref={scrollRef} />
            </div>
        </div>
    );
}
