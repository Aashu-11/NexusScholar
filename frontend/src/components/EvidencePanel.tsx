import React from 'react';
import { X, BookOpen, FileCheck, FileWarning } from 'lucide-react';
import type { EvidenceData } from '../types';
import { useEvidenceStore } from '../stores/evidenceStore';
import EvidenceRowCard from './EvidenceRow';

interface Props {
    evidenceData: EvidenceData | null;
}

export default function EvidencePanel({ evidenceData }: Props) {
    const { isOpen, activeTab, expandedCardId, highlightedCitation, close, setTab, toggleCard } = useEvidenceStore();

    if (!isOpen) return null;

    const rows = evidenceData?.rows || [];
    const confidence = evidenceData?.confidence ?? 0;
    const peerCount  = rows.filter(r => r.is_peer_reviewed).length;
    const prepCount  = rows.filter(r => !r.is_peer_reviewed).length;
    const confPct    = Math.round(confidence * 100);
    const confLevel  = confPct >= 65 ? 'high' : confPct >= 40 ? 'medium' : 'low';

    return (
        <aside className="evidence-panel">
            {/* Header */}
            <div className="evidence-header">
                <div className="evidence-header-title">
                    <BookOpen size={13} style={{ color: 'var(--ns-accent)' }} />
                    Sources
                    {rows.length > 0 && (
                        <span className="evidence-count-badge">{rows.length}</span>
                    )}
                </div>
                <button className="toggle-btn" onClick={close} title="Close panel">
                    <X size={15} />
                </button>
            </div>

            {/* Confidence summary */}
            {rows.length > 0 && (
                <div className="evidence-confidence-bar">
                    <div className="evidence-confidence-row">
                        <span className="evidence-confidence-label">Retrieval confidence</span>
                        <span className="evidence-confidence-value">{confPct}%</span>
                    </div>
                    <div className="confidence-track">
                        <div
                            className={`confidence-fill ${confLevel}`}
                            style={{ width: `${confPct}%` }}
                        />
                    </div>
                    <div className="evidence-stats-row">
                        <span className="evidence-stat-chip total">
                            <BookOpen size={9} /> {rows.length} total
                        </span>
                        {peerCount > 0 && (
                            <span className="evidence-stat-chip peer">
                                <FileCheck size={9} /> {peerCount} peer-reviewed
                            </span>
                        )}
                        {prepCount > 0 && (
                            <span className="evidence-stat-chip preprint">
                                <FileWarning size={9} /> {prepCount} preprint
                            </span>
                        )}
                    </div>
                </div>
            )}

            {/* Tabs */}
            <div className="evidence-tabs">
                <button
                    className={`evidence-tab ${activeTab === 'evidence' ? 'active' : ''}`}
                    onClick={() => setTab('evidence')}
                >
                    Sources
                </button>
                <button
                    className={`evidence-tab ${activeTab === 'discovery' ? 'active' : ''}`}
                    onClick={() => setTab('discovery')}
                >
                    Discovery
                </button>
            </div>

            {/* Content */}
            <div className="evidence-list">
                {activeTab === 'evidence' && rows.map((row, idx) => (
                    <EvidenceRowCard
                        key={row.evidence_id || idx}
                        row={row}
                        index={idx + 1}
                        isExpanded={expandedCardId === row.evidence_id}
                        isHighlighted={highlightedCitation === String(idx + 1)}
                        onToggle={() => toggleCard(row.evidence_id)}
                    />
                ))}
                {activeTab === 'evidence' && rows.length === 0 && (
                    <div className="evidence-empty">
                        Evidence sources will appear here once you ask a question.
                        <br />
                        <span style={{ fontSize: 11, color: 'var(--ns-text-muted)', display: 'block', marginTop: 6 }}>
                            Upload PDFs to build your corpus.
                        </span>
                    </div>
                )}
                {activeTab === 'discovery' && (
                    <div className="evidence-empty">
                        Citation graph exploration and paper discovery
                        <br />
                        <span style={{ fontSize: 11, color: 'var(--ns-text-muted)', display: 'block', marginTop: 6 }}>
                            Coming in Phase 2
                        </span>
                    </div>
                )}
            </div>
        </aside>
    );
}
