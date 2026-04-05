import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Send, Paperclip, Upload } from 'lucide-react';

interface Props {
    onSend: (query: string, recencyFilter?: string) => void;
    onUpload: (file: File) => void;
    isStreaming: boolean;
    currentIntent: string | null;
}

const RECENCY_OPTIONS = [
    { value: 'any', label: 'Any time' },
    { value: '3y',  label: 'Last 3 years' },
    { value: '1y',  label: 'Last year' },
];

export default function QueryInput({ onSend, onUpload, isStreaming, currentIntent }: Props) {
    const [query, setQuery] = useState('');
    const [isDragging, setIsDragging] = useState(false);
    const [recency, setRecency] = useState('any');
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const fileRef = useRef<HTMLInputElement>(null);

    // Auto-resize textarea
    useEffect(() => {
        const el = textareaRef.current;
        if (!el) return;
        el.style.height = 'auto';
        el.style.height = Math.min(el.scrollHeight, 180) + 'px';
    }, [query]);

    const send = useCallback(() => {
        if (query.trim() && !isStreaming) {
            onSend(query.trim(), recency);
            setQuery('');
        }
    }, [query, isStreaming, onSend, recency]);

    const onKey = useCallback((e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
    }, [send]);

    const onFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const f = e.target.files?.[0];
        if (f?.type === 'application/pdf') { onUpload(f); e.target.value = ''; }
    }, [onUpload]);

    const onDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        const f = e.dataTransfer.files?.[0];
        if (f?.type === 'application/pdf') onUpload(f);
    }, [onUpload]);

    const onDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    }, []);

    return (
        <>
            {isDragging && (
                <div className="drop-zone-overlay">
                    <div className="drop-zone-label">
                        <Upload size={20} />
                        Drop PDF to ingest into corpus
                    </div>
                </div>
            )}

            <div
                className="query-input-area"
                onDragOver={onDragOver}
                onDragLeave={() => setIsDragging(false)}
                onDrop={onDrop}
            >
                <div className="query-input-container">
                    <div className="query-input-wrapper">
                        <button
                            className="attach-btn"
                            onClick={() => fileRef.current?.click()}
                            title="Upload PDF to corpus"
                            disabled={isStreaming}
                        >
                            <Paperclip size={15} />
                        </button>
                        <input
                            ref={fileRef}
                            type="file"
                            accept=".pdf"
                            onChange={onFile}
                            style={{ display: 'none' }}
                        />
                        <textarea
                            ref={textareaRef}
                            className="query-textarea"
                            value={query}
                            onChange={e => setQuery(e.target.value)}
                            onKeyDown={onKey}
                            placeholder="Ask anything about the literature…"
                            rows={1}
                            disabled={isStreaming}
                        />
                        <div className="query-actions">
                            <button
                                className="send-btn"
                                onClick={send}
                                disabled={!query.trim() || isStreaming}
                                title="Send (Enter)"
                            >
                                <Send size={15} />
                            </button>
                        </div>
                    </div>

                    <div className="query-input-meta">
                        <span style={{ color: isStreaming ? 'var(--ns-accent)' : undefined }}>
                            {isStreaming ? 'Synthesizing answer…' : 'Enter to send · Shift+Enter for newline'}
                        </span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <select
                                className="recency-select"
                                value={recency}
                                onChange={e => setRecency(e.target.value)}
                                disabled={isStreaming}
                                title="Recency filter"
                            >
                                {RECENCY_OPTIONS.map(o => (
                                    <option key={o.value} value={o.value}>{o.label}</option>
                                ))}
                            </select>
                            {currentIntent && (
                                <span className="intent-pill">{currentIntent.replace(/_/g, ' ')}</span>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
}
