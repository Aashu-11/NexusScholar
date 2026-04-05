import React from 'react';
import { ShieldAlert } from 'lucide-react';

interface Props {
    content: string;
}

export default function AbstentionCard({ content }: Props) {
    const lines = content.split('\n');

    return (
        <div className="abstention-card">
            <div className="abstention-header">
                <ShieldAlert size={16} />
                <span>Evidence insufficient — abstaining</span>
            </div>
            <div className="abstention-body">
                {lines.map((line, i) => {
                    if (line.startsWith('## ') || line.startsWith('# '))
                        return <h3 key={i}>{line.replace(/^#+\s*/, '')}</h3>;
                    if (line.startsWith('**') && line.endsWith('**'))
                        return <p key={i}><strong>{line.replace(/\*\*/g, '')}</strong></p>;
                    if (line.startsWith('- '))
                        return <div key={i} className="abstention-item">{line.slice(2)}</div>;
                    if (line.startsWith('*') && line.endsWith('*'))
                        return <p key={i} className="abstention-note">{line.replace(/\*/g, '')}</p>;
                    if (line.trim())
                        return <p key={i}>{line}</p>;
                    return <div key={i} style={{ height: 4 }} />;
                })}
            </div>
        </div>
    );
}
