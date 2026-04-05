import React from 'react';

interface Row {
    method: string;
    core_idea?: string;
    dataset?: string;
    metric?: string;
    score?: string;
    venue?: string;
    year?: number;
}

interface Props {
    rows: Row[];
}

export default function ComparisonTable({ rows }: Props) {
    if (!rows || rows.length === 0) return null;

    const columns = (Object.keys(rows[0]) as Array<keyof Row>).filter(key => rows.some(row => row[key]));

    return (
        <div className="comparison-table-wrap">
            <table className="comparison-table">
                <thead>
                    <tr>
                        {columns.map(col => (
                            <th key={col}>{col.replace(/_/g, ' ')}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {rows.map((row, i) => (
                        <tr key={i}>
                            {columns.map(col => (
                                <td key={col}>{String(row[col] ?? '—')}</td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
