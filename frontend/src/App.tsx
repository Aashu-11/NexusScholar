import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen } from 'lucide-react';
import { useChatStore } from './stores/chatStore';
import { useEvidenceStore } from './stores/evidenceStore';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import EvidencePanel from './components/EvidencePanel';
import QueryInput from './components/QueryInput';

function App() {
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

    const {
        conversationId, conversations, messages,
        isStreaming, pipelineStatus, currentIntent,
        evidenceData, uploadStatus,
        newConversation, loadConversations, loadConversation,
        sendMessage, handleUpload,
    } = useChatStore();

    const {
        isOpen: evidenceOpen,
        open: openEvidence,
        toggle: toggleEvidence,
        highlightCitation,
    } = useEvidenceStore();

    useEffect(() => { loadConversations(); }, []);

    // Auto-open evidence panel when evidence data arrives
    useEffect(() => {
        if (evidenceData && evidenceData.rows.length > 0) openEvidence();
    }, [evidenceData]);

    const onCitationClick = (num: string) => {
        highlightCitation(num);
        openEvidence();
    };

    const handleSend = (query: string, recencyFilter?: string) => {
        sendMessage(query, recencyFilter);
    };

    return (
        <div className="app-layout">
            <Sidebar
                collapsed={sidebarCollapsed}
                conversations={conversations}
                activeId={conversationId}
                onNewChat={newConversation}
                onSelect={loadConversation}
            />

            <div className="main-chat">
                {/* Topbar */}
                <div className="chat-topbar">
                    <div className="topbar-left">
                        <button
                            className="toggle-sidebar-btn"
                            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                            title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                        >
                            {sidebarCollapsed
                                ? <PanelLeftOpen size={16} />
                                : <PanelLeftClose size={16} />
                            }
                        </button>

                        <div className="topbar-divider" />

                        <span className="topbar-brand">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                                stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <circle cx="12" cy="12" r="3" />
                                <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83
                                         M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                            </svg>
                            NexusScholar
                        </span>

                        {currentIntent && (
                            <span className="intent-pill">{currentIntent.replace(/_/g, ' ')}</span>
                        )}

                        {uploadStatus && (
                            <span style={{
                                fontSize: 11,
                                color: uploadStatus.startsWith('Failed')
                                    ? 'var(--ns-danger)'
                                    : 'var(--ns-success)',
                                marginLeft: 4,
                                letterSpacing: '-0.01em',
                            }}>
                                {uploadStatus}
                            </span>
                        )}
                    </div>

                    <div className="topbar-right">
                        <button
                            className="toggle-evidence-btn"
                            onClick={toggleEvidence}
                            title={evidenceOpen ? 'Hide sources panel' : 'Show sources panel'}
                        >
                            {evidenceOpen
                                ? <PanelRightClose size={16} />
                                : <PanelRightOpen size={16} />
                            }
                        </button>
                    </div>
                </div>

                <ChatArea
                    messages={messages}
                    pipelineStatus={pipelineStatus}
                    onSend={handleSend}
                    onCitationClick={onCitationClick}
                />

                <QueryInput
                    onSend={handleSend}
                    onUpload={handleUpload}
                    isStreaming={isStreaming}
                    currentIntent={currentIntent}
                />
            </div>

            <EvidencePanel evidenceData={evidenceData} />
        </div>
    );
}

createRoot(document.getElementById('root')!).render(<App />);
