import './lab.css';
import { LabProvider, useLab } from './lab/LabContext';
import { LabToolbar } from './lab/LabToolbar';
import { TermDrawer } from './lab/TermDrawer';
import { WelcomeModal } from './lab/WelcomeModal';
import LoraBasicsPage from './pages/LoraBasicsPage';
import { PlaceholderPage } from './pages/PlaceholderPage';
import SimulatorPage from './pages/SimulatorPage';

/**
 * FLoRIST Learning Lab: a four-page path through the language-models module.
 * 1 LoRA basics · 2 The aggregation problem · 3 FLoRIST step-by-step · 4 Simulator.
 */
export default function LanguageModelsPage() {
  return (
    <LabProvider>
      <LabShell />
    </LabProvider>
  );
}

function LabShell() {
  const { page, level } = useLab();
  return (
    <div className={`lm-lab ${page === 4 ? 'lm-wide' : ''}`}>
      <LabToolbar />
      {page === 1 && <LoraBasicsPage />}
      {page === 2 && (
        <PlaceholderPage
          number={2}
          title="The aggregation problem"
          lead="Why averaging $B$ and $A$ separately (FedIT) does not give the average of the clients' updates $B_k A_k$, and what exact stacking costs instead."
          planned={[
            'The cross terms and the $w_k^2$ diagonal weighting, shown on small matrices you can edit.',
            'Exact weighted stacking: correct, but the download rank grows to $\\sum_k r_k$.',
            'Heterogeneous ranks: why FedIT cannot average adapters of different sizes.',
          ]}
        />
      )}
      {page === 3 && (
        <PlaceholderPage
          number={3}
          title="FLoRIST step-by-step"
          lead="Every intermediate matrix of FLoRIST (Algorithm 1): the stacks, their thin SVDs, $Q$, $P$, the SVD of $P$, and the energy threshold $\tau$."
          planned={[
            'Step through the 22 named intermediate states computed by the backend.',
            'See the selected rank $p$ and the Eckart–Young error bound (Eq. 5).',
            'Rank matching: how each client pads or truncates the global adapters.',
          ]}
        />
      )}
      {page === 4 && <SimulatorPage />}
      <TermDrawer />
      {level === null && <WelcomeModal />}
    </div>
  );
}
