import React from 'react';

interface Props {
    supporting: number;
    contradicting: number;
    strength: string;
}

export default function ConsensusBar({ supporting, contradicting, strength }: Props) {
    if (supporting + contradicting < 2) return null;

    const strengthLabel: Record<string, string> = {
        broadly_supported: 'Broadly supported',
        contested: 'Contested',
        preliminary: 'Preliminary',
        insufficient: 'Insufficient evidence',
    };

    return (
        <span className={`consensus-badge strength-${strength}`}>
            <span className="cb-support">▲ {supporting} support</span>
            {contradicting > 0 && <span className="cb-contra"> · ▼ {contradicting} contradict</span>}
            <span className="cb-label"> — {strengthLabel[strength] || strength}</span>
        </span>
    );
}