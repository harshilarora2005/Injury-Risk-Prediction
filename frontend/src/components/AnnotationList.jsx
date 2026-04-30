const SIGNAL_LABEL = {
  knee:  "Reduced knee flexion",
  asym:  "Asymmetric loading",
  trunk: "Excessive trunk lean",
  vel:   "High angular velocity",
};

export default function AnnotationList({ events }) {
  return (
    <div className="card">
      <div className="label-tiny mb-1">Step 6</div>
      <h2 className="text-lg font-semibold mb-4">High-risk events</h2>
      {(!events || events.length === 0) ? (
        <div className="text-sm text-muted">No high-risk events detected in this clip.</div>
      ) : (
        <ul className="space-y-3">
          {events.map((ev, i) => (
            <li key={i} className="rounded-lg border border-border bg-panel2/60 p-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="text-sm font-medium text-ink">{ev.annotation}</div>
                  <div className="text-xs text-muted mt-1">
                    Frames {ev.start_frame}–{ev.end_frame}
                    {ev.peak_frame != null && <> · peak frame {ev.peak_frame}</>}
                  </div>
                </div>
                <span className="chip border-risk-high/40 text-risk-high bg-risk-high/10 uppercase">
                  {SIGNAL_LABEL[ev.dominant_subscore] ?? ev.dominant_subscore}
                </span>
              </div>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-4 text-[11px] text-muted leading-relaxed">
        Annotations identify the most prominent biomechanical signal per window — an approximation
        of the BiLSTM's temporal pattern, not the causal interaction itself.
      </p>
    </div>
  );
}
