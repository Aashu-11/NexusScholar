import React from 'react';
import { AlertOctagon } from 'lucide-react';
import type { EntityWarning } from '../types';

interface Props {
    warning: EntityWarning;
}

export default function EntityWarningBanner({ warning }: Props) {
    if (warning.entityCorrect) return null;

    return (
        <div className="entity-warning-banner">
            <div className="entity-warning-title">
                <AlertOctagon size={14} />
                Entity Mismatch Detected
                {warning.substitutedEntity && (
                    <span className="entity-warning-confidence">
                        ({Math.round(warning.confidence * 100)}% confidence)
                    </span>
                )}
            </div>
            {warning.substitutedEntity && (
                <p className="entity-warning-body">
                    Answer may describe <strong>{warning.substitutedEntity}</strong> instead of the requested entity.
                </p>
            )}
            {warning.issues && warning.issues.length > 0 && (
                <ul className="entity-warning-issues">
                    {warning.issues.map((issue, i) => <li key={i}>{issue}</li>)}
                </ul>
            )}
            <p className="entity-warning-note">
                Upload papers about the specific entity to improve grounding accuracy.
            </p>
        </div>
    );
}
