import { pct, riskColor } from "../lib/format";

function Cell({ label, value, klass }) {
  return (
    <div className={`rounded-lg border p-4 ${klass}`}>
      <div className="label-tiny opacity-80">{label}</div>
      <div className="mt-1 text-2xl font-semibold font-mono">{value}</div>
    </div>
  );
}

export default function RiskSummaryTable({ summary }) {
  if (!summary) return null;
  const rows = [
    ["Low",    summary.low_count,    summary.low_pct,    "Low"],
    ["Medium", summary.medium_count, summary.medium_pct, "Medium"],
    ["High",   summary.high_count,   summary.high_pct,   "High"],
  ];
  return (
    <div className="card">
      <div className="label-tiny mb-1">Step 2</div>
      <h2 className="text-lg font-semibold mb-4">Risk distribution</h2>
      <div className="grid grid-cols-3 gap-3 mb-4">
        {rows.map(([label, n, p, key]) => (
          <Cell key={key} label={label} value={`${n}`} klass={riskColor(key)} />
        ))}
      </div>
      <table className="w-full text-sm">
        <thead className="text-muted">
          <tr className="text-left border-b border-border">
            <th className="py-2 font-medium">Level</th>
            <th className="py-2 font-medium text-right">Windows</th>
            <th className="py-2 font-medium text-right">Share</th>
          </tr>
        </thead>
        <tbody className="font-mono">
          {rows.map(([label, n, p]) => (
            <tr key={label} className="border-b border-border/60">
              <td className="py-2">{label}</td>
              <td className="py-2 text-right">{n}</td>
              <td className="py-2 text-right">{pct(p)}</td>
            </tr>
          ))}
          <tr>
            <td className="py-2 font-semibold">Total</td>
            <td className="py-2 text-right font-semibold">{summary.total_windows}</td>
            <td className="py-2 text-right font-semibold">100.0%</td>
          </tr>
        </tbody>
      </table>

      {summary.peak_high_window && (
        <div className="mt-4 rounded-lg border border-risk-high/40 bg-risk-high/5 p-3 text-sm">
          <span className="label-tiny text-risk-high">Peak risk window</span>
          <div className="mt-1 font-mono">
            Frames {summary.peak_high_window.start_frame}–{summary.peak_high_window.end_frame}
            {" · "}P(High) = {summary.peak_high_window.P_high.toFixed(3)}
          </div>
        </div>
      )}
    </div>
  );
}
