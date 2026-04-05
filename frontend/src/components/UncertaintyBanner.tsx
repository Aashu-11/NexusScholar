import React from 'react';
import { AlertTriangle } from 'lucide-react';

interface Props {
    flags: string[];
}

export default function UncertaintyBanner({ flags }: Props) {
    if (!flags || flags.length === 0) return null;

    return (
        <div className="uncertainty-banner">
            <AlertTriangle size={13} style={{ flexShrink: 0, marginTop: 1 }} />
            <div style={{ lineHeight: 1.5 }}>
                {flags.map((f, i) => <div key={i}>{f}</div>)}
            </div>
        </div>
    );
}
