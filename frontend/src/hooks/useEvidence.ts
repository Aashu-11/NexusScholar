/* Hook for evidence panel data + actions */
import { useChatStore } from '../stores/chatStore';
import { useEvidenceStore } from '../stores/evidenceStore';

export function useEvidence() {
    const evidenceData = useChatStore(s => s.evidenceData);
    const panel = useEvidenceStore();

    const onCitationClick = (num: string) => {
        panel.highlightCitation(num);
        panel.open();
        panel.setTab('evidence');
    };

    return { evidenceData, ...panel, onCitationClick };
}