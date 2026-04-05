import React from 'react';
import { Plus } from 'lucide-react';
import type { Conversation } from '../types';

interface Props {
    collapsed: boolean;
    conversations: Conversation[];
    activeId: string;
    onNewChat: () => void;
    onSelect: (id: string) => void;
}

export default function Sidebar({ collapsed, conversations, activeId, onNewChat, onSelect }: Props) {
    const grouped = groupByRecency(conversations);

    return (
        <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
            <div className="sidebar-header">
                <div className="sidebar-brand">
                    <div className="sidebar-brand-icon">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="12" cy="12" r="3" />
                            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83
                                     M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                        </svg>
                    </div>
                    <span className="sidebar-brand-name">NexusScholar</span>
                    <span className="sidebar-brand-badge">Beta</span>
                </div>

                <button className="new-chat-btn" onClick={onNewChat}>
                    <Plus size={14} />
                    New conversation
                </button>
            </div>

            <div className="sidebar-conversations">
                {Object.entries(grouped).map(([group, convs]) => (
                    <div key={group}>
                        <div className="conv-group-label">{group}</div>
                        {convs.map(c => (
                            <div
                                key={c.conversation_id}
                                className={`conv-item ${c.conversation_id === activeId ? 'active' : ''}`}
                                onClick={() => onSelect(c.conversation_id)}
                                title={c.title || 'New conversation'}
                            >
                                {c.title || 'New conversation'}
                            </div>
                        ))}
                    </div>
                ))}
                {conversations.length === 0 && (
                    <div className="sidebar-empty">No conversations yet.<br />Ask your first research question.</div>
                )}
            </div>

            <div className="sidebar-footer">
                Evidence-first research intelligence
            </div>
        </aside>
    );
}

function groupByRecency(convs: Conversation[]) {
    const now = Date.now();
    const groups: Record<string, Conversation[]> = {
        Today: [], Yesterday: [], 'Last 7 days': [], Earlier: [],
    };

    convs.forEach(c => {
        const diff = Math.floor((now - new Date(c.updated_at || c.created_at).getTime()) / 86_400_000);
        if (diff === 0)      groups['Today'].push(c);
        else if (diff === 1) groups['Yesterday'].push(c);
        else if (diff <= 7)  groups['Last 7 days'].push(c);
        else                 groups['Earlier'].push(c);
    });

    return Object.fromEntries(Object.entries(groups).filter(([, v]) => v.length > 0));
}
