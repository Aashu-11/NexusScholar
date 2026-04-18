import React, { useContext, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
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

// ── Citation chip helpers ──────────────────────────────────────────────────────

const CITE_PATTERN = /(\[\d+(?:,\s*\d+)*\])/g;

function splitWithCitations(
    text: string,
    onCitationClick: ((n: string) => void) | undefined,
    baseKey: number,
): React.ReactNode[] {
    const parts = text.split(CITE_PATTERN);
    const nodes: React.ReactNode[] = [];
    let k = baseKey;

    for (const part of parts) {
        if (!part) continue;
        const m = part.match(/^\[(\d+(?:,\s*\d+)*)\]$/);
        if (m) {
            m[1].split(',').forEach(raw => {
                const num = raw.trim();
                nodes.push(
                    <span
                        key={k++}
                        className="citation-chip"
                        onClick={() => onCitationClick?.(num)}
                        title={`Source [${num}]`}
                    >
                        {num}
                    </span>
                );
            });
        } else {
            nodes.push(<React.Fragment key={k++}>{part}</React.Fragment>);
        }
    }
    return nodes;
}

/** Recursively processes React children, injecting citation chips into string nodes. */
function withCitations(
    children: React.ReactNode,
    onCitationClick: ((n: string) => void) | undefined,
    depth = 0,
): React.ReactNode {
    let k = depth * 10000;
    return React.Children.map(children, child => {
        if (typeof child === 'string') {
            return CITE_PATTERN.test(child)
                ? splitWithCitations(child, onCitationClick, k++)
                : child;
        }
        if (React.isValidElement(child)) {
            const el = child as React.ReactElement<{ children?: React.ReactNode }>;
            if (el.props.children != null) {
                return React.cloneElement(el, {
                    children: withCitations(el.props.children, onCitationClick, depth + 1),
                } as Partial<typeof el.props>);
            }
        }
        return child;
    });
}

// ── List context (ordered vs unordered) ──────────────────────────────────────

const ListContext = React.createContext<boolean>(false);

// ── Renderer ──────────────────────────────────────────────────────────────────

interface RenderedMarkdownProps {
    text: string;
    isStreaming?: boolean;
    onCitationClick?: (n: string) => void;
}

const REMARK_PLUGINS = [remarkMath];
const REHYPE_PLUGINS = [rehypeKatex];

function RenderedMarkdown({ text, isStreaming, onCitationClick }: RenderedMarkdownProps) {
    const components = useMemo(() => buildComponents(onCitationClick), [onCitationClick]);

    return (
        <div className="md-body">
            <ReactMarkdown
                remarkPlugins={REMARK_PLUGINS}
                rehypePlugins={REHYPE_PLUGINS}
                components={components}
            >
                {text || ''}
            </ReactMarkdown>
            {isStreaming && text && <span className="streaming-cursor" />}
        </div>
    );
}

// Build the components map once per onCitationClick reference change.
function buildComponents(onCitationClick: ((n: string) => void) | undefined) {
    const cite = (children: React.ReactNode) => withCitations(children, onCitationClick);

    return {
        // ── Headings ──────────────────────────────────────────────────────
        h1: ({ children }: { children?: React.ReactNode }) =>
            <h1 className="md-h1">{cite(children)}</h1>,

        h2: ({ children }: { children?: React.ReactNode }) =>
            <h2 className="md-h2">{cite(children)}</h2>,

        h3: ({ children }: { children?: React.ReactNode }) =>
            <h3 className="md-h3">{cite(children)}</h3>,

        h4: ({ children }: { children?: React.ReactNode }) =>
            <h3 className="md-h3">{cite(children)}</h3>,

        // ── Paragraph ─────────────────────────────────────────────────────
        p: ({ children }: { children?: React.ReactNode }) =>
            <p className="md-p">{cite(children)}</p>,

        // ── Blockquote ────────────────────────────────────────────────────
        blockquote: ({ children }: { children?: React.ReactNode }) =>
            <blockquote className="md-blockquote">{cite(children)}</blockquote>,

        // ── Lists ─────────────────────────────────────────────────────────
        ul: ({ children }: { children?: React.ReactNode }) => (
            <ListContext.Provider value={false}>
                <div>{children}</div>
            </ListContext.Provider>
        ),

        ol: ({ children }: { children?: React.ReactNode }) => (
            <ListContext.Provider value={true}>
                <OrderedList>{children}</OrderedList>
            </ListContext.Provider>
        ),

        li: LiItem,

        // ── Horizontal rule ───────────────────────────────────────────────
        hr: () => <hr className="md-hr" />,

        // ── Code ─────────────────────────────────────────────────────────
        // react-markdown v10: override `pre` for fenced code blocks
        pre: ({ children }: { children?: React.ReactNode }) => {
            const codeEl = React.Children.toArray(children).find(
                (c): c is React.ReactElement<{ className?: string; children?: React.ReactNode }> =>
                    React.isValidElement(c)
            ) as React.ReactElement<{ className?: string; children?: React.ReactNode }> | undefined;

            const lang = codeEl?.props.className?.replace(/^language-/, '') ?? '';
            const code = codeEl?.props.children ?? children;

            return (
                <div className="md-codeblock">
                    {lang && (
                        <div className="md-codeblock-header">
                            <span className="md-codeblock-lang">{lang}</span>
                        </div>
                    )}
                    <pre><code>{code}</code></pre>
                </div>
            );
        },

        // `code` here handles only inline code (block code is caught by `pre`)
        code: ({ children, className }: { children?: React.ReactNode; className?: string }) => {
            // If it has a language class it's a block code node inside our pre — render plainly
            if (className?.startsWith('language-')) {
                return <code className={className}>{children}</code>;
            }
            return <code>{children}</code>;
        },

        // ── Table ─────────────────────────────────────────────────────────
        table: ({ children }: { children?: React.ReactNode }) => (
            <div className="md-table-wrap">
                <table className="md-table">{children}</table>
            </div>
        ),

        th: ({ children }: { children?: React.ReactNode }) =>
            <th>{cite(children)}</th>,

        td: ({ children }: { children?: React.ReactNode }) =>
            <td>{cite(children)}</td>,
    } as Parameters<typeof ReactMarkdown>[0]['components'];
}

// ── Ordered list wrapper with counter ────────────────────────────────────────

const OrderedCountContext = React.createContext<{ next: () => number }>({ next: () => 0 });

function OrderedList({ children }: { children?: React.ReactNode }) {
    const counter = React.useRef(0);
    const ctx = useMemo(() => ({
        next: () => { counter.current += 1; return counter.current; },
    }), []);

    // Reset counter on each render pass
    counter.current = 0;

    return (
        <OrderedCountContext.Provider value={ctx}>
            <div>{children}</div>
        </OrderedCountContext.Provider>
    );
}

function LiItem({ children }: { children?: React.ReactNode }) {
    const ordered = useContext(ListContext);
    const { next } = useContext(OrderedCountContext);
    const num = ordered ? next() : 0;

    return (
        <div className="md-li">
            {ordered
                ? <span className="md-li-num">{num}.</span>
                : <span className="md-li-bullet">›</span>
            }
            <span>{children}</span>
        </div>
    );
}
