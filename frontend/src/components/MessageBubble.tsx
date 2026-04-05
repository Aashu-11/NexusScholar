import React, { useMemo } from 'react';
import { BookOpen, FileCheck, FileWarning } from 'lucide-react';
import type { Message } from '../types';
import UncertaintyBanner from './UncertaintyBanner';
import AbstentionCard from './AbstentionCard';
import EntityWarningBanner from './EntityWarningBanner';

interface Props {
    message: Message;
    onCitationClick?: (num: string) => void;
}

export default function MessageBubble({ message, onCitationClick }: Props) {
    if (message.role === 'user') {
        return (
            <div className="message message-user">
                <div className="message-content">{message.content}</div>
            </div>
        );
    }

    const isAbstention = message.isAbstention && message.content.includes('Insufficient Evidence');

    return (
        <div className="message message-assistant">
            <div className="assistant-avatar">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="3" />
                    <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83
                             M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                </svg>
            </div>

            <div className="message-body">
                {message.uncertaintyFlags && message.uncertaintyFlags.length > 0 && (
                    <UncertaintyBanner flags={message.uncertaintyFlags} />
                )}
                {message.entityWarning && !message.entityWarning.entityCorrect && (
                    <EntityWarningBanner warning={message.entityWarning} />
                )}

                {isAbstention ? (
                    <AbstentionCard content={message.content} />
                ) : (
                    <div className="message-content">
                        <RenderedMarkdown
                            text={message.content}
                            isStreaming={message.isStreaming}
                            onCitationClick={onCitationClick}
                        />
                    </div>
                )}

                {!message.isStreaming && (message.totalSources ?? 0) > 0 && (
                    <div className="source-badges">
                        <span className="source-badge total">
                            <BookOpen size={11} /> {message.totalSources} sources
                        </span>
                        {(message.peerReviewedCount ?? 0) > 0 && (
                            <span className="source-badge peer-reviewed">
                                <FileCheck size={11} /> {message.peerReviewedCount} peer-reviewed
                            </span>
                        )}
                        {(message.preprintCount ?? 0) > 0 && (
                            <span className="source-badge preprint">
                                <FileWarning size={11} /> {message.preprintCount} preprints
                            </span>
                        )}
                    </div>
                )}

                {message.isStreaming && !message.content && (
                    <span className="streaming-dots">
                        <span /><span /><span />
                    </span>
                )}
            </div>
        </div>
    );
}

// ── Markdown parser types ──────────────────────────────────────────────────────

type Seg =
    | { type: 'h1' | 'h2' | 'h3'; content: string }
    | { type: 'p' | 'blockquote'; content: string }
    | { type: 'li'; content: string; ordered: boolean; num: number }
    | { type: 'hr' | 'spacer' }
    | { type: 'table'; headers: string[]; rows: string[][] }
    | { type: 'codeblock'; lang: string; content: string };

// ── Parser ─────────────────────────────────────────────────────────────────────

function parseMarkdown(text: string): Seg[] {
    const lines = text.split('\n');
    const segs: Seg[] = [];
    let i = 0;

    while (i < lines.length) {
        const line = lines[i];
        const t = line.trim();

        // ── Table block ────────────────────────────────────────────
        if (t.startsWith('|')) {
            const tLines: string[] = [];
            while (i < lines.length && lines[i].trim().startsWith('|')) {
                tLines.push(lines[i].trim());
                i++;
            }
            const tSeg = parseTable(tLines);
            if (tSeg) segs.push(tSeg);
            continue;
        }

        // ── Fenced code block ──────────────────────────────────────
        if (t.startsWith('```')) {
            const lang = t.slice(3).trim();
            const code: string[] = [];
            i++;
            while (i < lines.length && !lines[i].trim().startsWith('```')) {
                code.push(lines[i]);
                i++;
            }
            if (i < lines.length) i++; // consume closing ```
            segs.push({ type: 'codeblock', lang, content: code.join('\n') });
            continue;
        }

        // ── Headings ───────────────────────────────────────────────
        if (line.startsWith('#### ')) { segs.push({ type: 'h3', content: line.slice(5) }); i++; continue; }
        if (line.startsWith('### '))  { segs.push({ type: 'h3', content: line.slice(4) }); i++; continue; }
        if (line.startsWith('## '))   { segs.push({ type: 'h2', content: line.slice(3) }); i++; continue; }
        if (line.startsWith('# '))    { segs.push({ type: 'h1', content: line.slice(2) }); i++; continue; }

        // ── Blockquote ─────────────────────────────────────────────
        if (line.startsWith('> ')) { segs.push({ type: 'blockquote', content: line.slice(2) }); i++; continue; }

        // ── Horizontal rule ────────────────────────────────────────
        if (/^[-*_]{3,}$/.test(t)) { segs.push({ type: 'hr' }); i++; continue; }

        // ── Unordered list item ────────────────────────────────────
        const ulm = line.match(/^(\s*)[-*+]\s(.*)$/);
        if (ulm) {
            segs.push({ type: 'li', content: ulm[2], ordered: false, num: 0 });
            i++; continue;
        }

        // ── Ordered list item ──────────────────────────────────────
        const olm = line.match(/^(\s*)(\d+)\.\s(.*)$/);
        if (olm) {
            segs.push({ type: 'li', content: olm[3], ordered: true, num: parseInt(olm[2], 10) });
            i++; continue;
        }

        // ── Empty line ─────────────────────────────────────────────
        if (!t) { segs.push({ type: 'spacer' }); i++; continue; }

        // ── Regular paragraph ──────────────────────────────────────
        segs.push({ type: 'p', content: line });
        i++;
    }

    return segs;
}

function parseTable(tableLines: string[]): Seg | null {
    if (tableLines.length < 2) return null;

    const splitCells = (line: string): string[] => {
        const parts = line.split('|');
        // Trim surrounding empty strings from leading/trailing |
        const start = parts[0].trim() === '' ? 1 : 0;
        const end = parts[parts.length - 1].trim() === '' ? parts.length - 1 : parts.length;
        return parts.slice(start, end).map(c => c.trim());
    };

    const isSep = (line: string) => /^[\s|:\-]+$/.test(line);

    // Need at least header + separator rows
    if (!isSep(tableLines[1])) return null;

    const headers = splitCells(tableLines[0]);
    const rows = tableLines.slice(2).map(splitCells).filter(r => r.length > 0);

    return { type: 'table', headers, rows };
}

// ── Renderer ───────────────────────────────────────────────────────────────────

interface RenderedMarkdownProps {
    text: string;
    isStreaming?: boolean;
    onCitationClick?: (n: string) => void;
}

function RenderedMarkdown({ text, isStreaming, onCitationClick }: RenderedMarkdownProps) {
    const segs = useMemo(() => parseMarkdown(text || ''), [text]);

    return (
        <div className="md-body">
            {segs.map((seg, idx) => (
                <SegmentView key={idx} seg={seg} onCitationClick={onCitationClick} />
            ))}
            {isStreaming && text && <span className="streaming-cursor" />}
        </div>
    );
}

function SegmentView({ seg, onCitationClick }: { seg: Seg; onCitationClick?: (n: string) => void }) {
    switch (seg.type) {
        case 'h1':
            return <h1 className="md-h1">{inlineRender(seg.content, onCitationClick)}</h1>;
        case 'h2':
            return <h2 className="md-h2">{inlineRender(seg.content, onCitationClick)}</h2>;
        case 'h3':
            return <h3 className="md-h3">{inlineRender(seg.content, onCitationClick)}</h3>;
        case 'p':
            return <p className="md-p">{inlineRender(seg.content, onCitationClick)}</p>;
        case 'blockquote':
            return <blockquote className="md-blockquote">{inlineRender(seg.content, onCitationClick)}</blockquote>;
        case 'li':
            return (
                <div className="md-li">
                    {seg.ordered
                        ? <span className="md-li-num">{seg.num}.</span>
                        : <span className="md-li-bullet">›</span>
                    }
                    <span>{inlineRender(seg.content, onCitationClick)}</span>
                </div>
            );
        case 'hr':
            return <hr className="md-hr" />;
        case 'spacer':
            return <div className="md-spacer" />;
        case 'codeblock':
            return (
                <div className="md-codeblock">
                    {seg.lang && (
                        <div className="md-codeblock-header">
                            <span className="md-codeblock-lang">{seg.lang}</span>
                        </div>
                    )}
                    <pre><code>{seg.content}</code></pre>
                </div>
            );
        case 'table':
            return <MarkdownTable headers={seg.headers} rows={seg.rows} onCitationClick={onCitationClick} />;
        default:
            return null;
    }
}

function MarkdownTable({ headers, rows, onCitationClick }: {
    headers: string[];
    rows: string[][];
    onCitationClick?: (n: string) => void;
}) {
    // Detect numeric columns for right-alignment
    const isNumericCol = headers.map((_, ci) =>
        rows.every(r => {
            const v = (r[ci] || '').replace(/[%,]/g, '').trim();
            return v === '' || v === '—' || !isNaN(parseFloat(v));
        })
    );

    return (
        <div className="md-table-wrap">
            <table className="md-table">
                <thead>
                    <tr>
                        {headers.map((h, i) => (
                            <th key={i} className={isNumericCol[i] ? 'num' : ''}>
                                {inlineRender(h, onCitationClick)}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {rows.map((row, ri) => (
                        <tr key={ri}>
                            {headers.map((_, ci) => (
                                <td key={ci} className={isNumericCol[ci] ? 'num' : ''}>
                                    {inlineRender(row[ci] ?? '', onCitationClick)}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

// ── Inline renderer ─────────────────────────────────────────────────────────────
// Handles: **bold**, *italic*, `code`, [N] citation chips, plain text

type InlineNode = React.ReactNode;

function inlineRender(text: string, onCitationClick?: (n: string) => void): InlineNode[] {
    if (!text) return [];

    // Tokenize by citation chips [N], **bold**, *italic*, `code`
    const pattern = /(\[\d+(?:,\s*\d+)*\]|\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
    const tokens = text.split(pattern);

    const result: InlineNode[] = [];
    let key = 0;

    for (const tok of tokens) {
        if (!tok) continue;

        // Citation chip [1] or [1,2,3]
        const citMatch = tok.match(/^\[(\d+(?:,\s*\d+)*)\]$/);
        if (citMatch) {
            citMatch[1].split(',').forEach(n => {
                const num = n.trim();
                result.push(
                    <span
                        key={key++}
                        className="citation-chip"
                        onClick={() => onCitationClick?.(num)}
                        title={`Source [${num}]`}
                    >
                        {num}
                    </span>
                );
            });
            continue;
        }

        // Bold **text**
        if (tok.startsWith('**') && tok.endsWith('**') && tok.length > 4) {
            result.push(<strong key={key++}>{tok.slice(2, -2)}</strong>);
            continue;
        }

        // Italic *text*
        if (tok.startsWith('*') && tok.endsWith('*') && tok.length > 2) {
            result.push(<em key={key++}>{tok.slice(1, -1)}</em>);
            continue;
        }

        // Inline code `text`
        if (tok.startsWith('`') && tok.endsWith('`') && tok.length > 2) {
            result.push(<code key={key++}>{tok.slice(1, -1)}</code>);
            continue;
        }

        result.push(<React.Fragment key={key++}>{tok}</React.Fragment>);
    }

    return result;
}
