import { formatInt, formatNumber } from '../../../components/format';
import type { SimulatorRoundResult, Table4Row } from '../types';
import { classesHeld } from './experiments';

/* Researcher-level tables: the raw numbers behind the charts. */

export function SingularValuesTable({ r }: { r: SimulatorRoundResult }) {
  const total = r.singular_values.reduce((a, s) => a + s * s, 0) || 1;
  return (
    <div className="lm-table-wrap">
      <table className="lm-table">
        <caption className="small muted">σ_i of P (equal to the singular values of ΔW)</caption>
        <thead>
          <tr>
            <th scope="col">i</th>
            <th scope="col">σ_i</th>
            <th scope="col">σ_i² / Σσ²</th>
            <th scope="col">cumulative</th>
            <th scope="col">kept</th>
          </tr>
        </thead>
        <tbody>
          {r.singular_values.map((s, i) => (
            <tr key={i}>
              <th scope="row">{i + 1}</th>
              <td>{s.toPrecision(4)}</td>
              <td>{formatNumber(((s * s) / total) * 100, 2)}%</td>
              <td>{formatNumber(r.cumulative_energy[i] * 100, 2)}%</td>
              <td>{i < r.p ? 'yes' : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ClientsTable({ r }: { r: SimulatorRoundResult }) {
  return (
    <div className="lm-table-wrap">
      <table className="lm-table">
        <caption className="small muted">Sampled clients this round</caption>
        <thead>
          <tr>
            <th scope="col">client k</th>
            <th scope="col">n_k</th>
            <th scope="col">w_k = n_k / N</th>
            <th scope="col">r_k</th>
            <th scope="col">classes held</th>
          </tr>
        </thead>
        <tbody>
          {r.sampled_clients.map((c, i) => (
            <tr key={c}>
              <th scope="row">#{c + 1}</th>
              <td>{formatInt(r.sample_counts[i])}</td>
              <td>{r.weights[i].toFixed(4)}</td>
              <td>{r.client_ranks[i]}</td>
              <td>{classesHeld(r.class_counts[c])} / 16</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ServerCostTable({ r, table4 }: { r: SimulatorRoundResult; table4?: { reference: string; caption: string; rows: Table4Row[] } }) {
  const sc = r.server_cost;
  return (
    <div className="lm-stack">
      <div className="lm-table-wrap">
        <table className="lm-table">
          <caption className="small muted">Server operations this round (estimated; see assumptions)</caption>
          <thead>
            <tr>
              <th scope="col">Method</th>
              <th scope="col">ops</th>
            </tr>
          </thead>
          <tbody>
            {sc.fedit_ops != null && (
              <tr>
                <th scope="row">FedIT: 2r(m+n)</th>
                <td>{formatInt(sc.fedit_ops)}</td>
              </tr>
            )}
            <tr>
              <th scope="row">FLoRIST (Algorithm 1)</th>
              <td>{formatInt(sc.florist_ops)}</td>
            </tr>
            <tr>
              <th scope="row">SVD of the full ΔW</th>
              <td>{formatInt(sc.direct_svd_ops)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="small muted">
        Σr_k = {sc.stacked_rank} {sc.stacked_rank_below_min_dim ? '<' : '≥'} min(m, n) = {Math.min(r.m, r.n)}:{' '}
        {sc.stacked_rank_below_min_dim ? 'the compact computation is cheaper than an SVD of ΔW.' : 'the compact-computation advantage does not apply.'}
      </p>
      {table4 && (
        <div className="lm-table-wrap lm-paper">
          <table className="lm-table">
            <caption className="small muted">
              <span className="badge">Paper-reported</span> {table4.reference}: {table4.caption}
            </caption>
            <thead>
              <tr>
                <th scope="col">Method</th>
                <th scope="col">server FLOPs</th>
              </tr>
            </thead>
            <tbody>
              {table4.rows.map((row) => (
                <tr key={row.method}>
                  <th scope="row">{row.method}</th>
                  <td>{row.server_flops === 0 ? '0' : row.server_flops.toExponential(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
