import React from 'react';
import { ChevronDown, ChevronUp, ExternalLink, FileText, AlertTriangle } from 'lucide-react';
import type { EvidenceRowData } from '../types';

interface Props {
    row: EvidenceRowData;
    index: number;
    isExpanded: boolean;
    isHighlighted: boolean;
    onToggle: () => void;
}

export default function EvidenceRowCard({ row, index, isExpanded, isHighlighted, onToggle }: Props) {
    const pct = Math.min(Math.round((row.relevance ?? 0) * 100), 100);
    const sourceUrl = row.source_url ?? '';
    const pdfUrl    = row.pdf_url ?? '';

    return (
        <div
            className={`evidence-card ${isHighlighted ? 'highlighted' : ''}`}
            onClick={onToggle}
        >
            <div className="evidence-card-header">
                <div className="evidence-card-title-row">
                    <span className="evidence-card-num">{index}</span>
                    <span className="evidence-card-title">{row.paper_title}</span>
                </div>
                {isExpanded
                    ? <ChevronUp size={13} color="var(--ns-text-tertiary)" style={{ flexShrink: 0 }} />
                    : <ChevronDown size={13} color="var(--ns-text-tertiary)" style={{ flexShrink: 0 }} />
                }
            </div>

            <div className="evidence-card-meta">
                {[row.authors, row.year && String(row.year), row.venue, row.section]
                    .filter(Boolean)
                    .join(' · ')}
            </div>

            <div className="evidence-tags">
                {row.is_peer_reviewed
                    ? <span className="evidence-tag peer-reviewed">Peer-reviewed</span>
                    : <span className="evidence-tag preprint">Preprint</span>
                }
                {row.is_retracted && (
                    <span className="evidence-tag retracted">
                        <AlertTriangle size={9} /> Retracted
                    </span>
                )}
            </div>

            <div className={`evidence-card-passage ${isExpanded ? 'expanded' : ''}`}>
                {row.passage_preview || row.chunk_text || ''}
            </div>

            <div className="relevance-bar">
                <div className="relevance-bar-fill" style={{ width: `${pct}%` }} />
            </div>

            {isExpanded && (
                <div className="evidence-card-actions">
                    {sourceUrl && (
                        <a
                            href={sourceUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="ev-action-btn"
                            onClick={e => e.stopPropagation()}
                        >
                            <ExternalLink size={10} /> View source
                        </a>
                    )}
                    {pdfUrl && (
                        <a
                            href={pdfUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="ev-action-btn"
                            onClick={e => e.stopPropagation()}
                        >
                            <FileText size={10} /> PDF
                        </a>
                    )}
                </div>
            )}
        </div>
    );
}
