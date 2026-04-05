import React, { useState } from 'react';
import type { CitationCard } from '../types';

interface Props {
    number: string;
    card?: CitationCard;
    onClick?: (num: string) => void;
}

export default function CitationChip({ number, card, onClick }: Props) {
    const [hovered, setHovered] = useState(false);

    return (
        <span className="citation-chip-wrap">
            <span
                className="citation-chip"
                onClick={() => onClick?.(number)}
                onMouseEnter={() => setHovered(true)}
                onMouseLeave={() => setHovered(false)}
                title={card ? `${card.paper_title} (${card.year})` : `Source [${number}]`}
            >
                {number}
            </span>
            {hovered && card && (
                <div className="citation-hover-card">
                    <div className="chc-title">{card.paper_title}</div>
                    <div className="chc-meta">
                        {card.authors}{card.year ? ` · ${card.year}` : ''}{card.venue ? ` · ${card.venue}` : ''}
                    </div>
                    <div className="chc-passage">{card.passage_preview}</div>
                    {card.is_retracted && <div className="chc-retracted">⚠ Retracted</div>}
                    {!card.is_peer_reviewed && <div className="chc-preprint">Preprint — not peer reviewed</div>}
                </div>
            )}
        </span>
    );
}