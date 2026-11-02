import { formatInt, formatNumber } from '../../../components/format';
import type { SimulatorRoundResult, Table4Row } from '../types';
import { M } from '../lab/Math';
import { classesHeld } from './experiments';

/** A number in scientific notation as TeX (1.99e-32 → 1.99 × 10⁻³²); plain for ordinary magnitudes. */
function num(x: number, digits = 4): string {
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1e-3 && a < 1e6) return x.toPrecision(digits);
  const [mant, exp] = x.toExponential(digits - 1).split('e');
  return String.raw`${mant} \times 10^{${Number(exp)}}`;
}

/* Researcher-level tables: the raw numbers behind the charts. */

export function SingularValuesTable({ r }: { r: SimulatorRoundResult }) {
  const total = r.singular_values.reduce((a, s) => a + s * s, 0) || 1;
  return (
    <div className="lm-table-wrap">
      <table className="lm-table">
        <caption className="small muted">
          Singular values <M tex="\sigma_i" /> of <M tex="P" /> (equal to the singular values of <M tex="\Delta W" />)
        </caption>
        <thead>
          <tr>
            <th scope="col">
              <M tex="i" />
            </th>
            <th scope="col">
              <M tex="\sigma_i" />
            </th>
            <th scope="col">
              <M tex="\sigma_i^2 / \sum_j \sigma_j^2" />
            </th>
            <th scope="col">cumulative</th>
            <th scope="col">kept</th>
          </tr>
        </thead>
        <tbody>
          {r.singular_values.map((s, i) => (
            <tr key={i}>
              <th scope="row">{i + 1}</th>
              <td>
                <M tex={num(s)} />
              </td>
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
            <th scope="col">
              client <M tex="k" />
            </th>
            <th scope="col">
              <M tex="n_k" />
            </th>
            <th scope="col">
              <M tex="w_k = n_k / N" />
            </th>
            <th scope="col">
              <M tex="r_k" />
            </th>
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
                <th scope="row">
                  FedIT: <M tex="2r(m+n)" />
                </th>
                <td>{formatInt(sc.fedit_ops)}</td>
              </tr>
            )}
            <tr>
              <th scope="row">FLoRIST (Algorithm 1)</th>
              <td>{formatInt(sc.florist_ops)}</td>
            </tr>
            <tr>
              <th scope="row">
                SVD of the full <M tex="\Delta W" />
              </th>
              <td>{formatInt(sc.direct_svd_ops)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="small muted">
        <M tex={String.raw`\sum_k r_k = ${sc.stacked_rank} ${sc.stacked_rank_below_min_dim ? '<' : '\\ge'} \min(m, n) = ${Math.min(r.m, r.n)}`} />:{' '}
        {sc.stacked_rank_below_min_dim ? (
          <>
            the compact computation is cheaper than an SVD of <M tex="\Delta W" />.
          </>
        ) : (
          'the compact-computation advantage does not apply.'
        )}
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
                  <td>
                    <M tex={num(row.server_flops, 3)} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
