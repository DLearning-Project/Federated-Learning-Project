import { lazy, type ComponentType, type LazyExoticComponent } from 'react';

/** Module registry: add a module here to get a route, nav link and homepage card. */
export interface ModuleDef {
  id: string;
  title: string;
  shortTitle: string;
  path: string;
  summary: string;
  topics: string[];
  owner: string;
  accent: string;
  Page: LazyExoticComponent<ComponentType>;
}

export const MODULES: ModuleDef[] = [
  {
    id: 'language-models',
    title: 'Language models',
    shortTitle: 'Language models',
    path: '/language-models',
    summary: 'Fine-tune a frozen language model with small LoRA adapters, and see how much less data each client has to send.',
    topics: ['LoRA', 'FLoRIST', 'Communication cost'],
    owner: 'Teammate A',
    accent: 'var(--accent-language-models)',
    Page: lazy(() => import('../modules/language-models/LanguageModelsPage')),
  },
  {
    id: 'acoustics',
    title: 'Underwater acoustics',
    shortTitle: 'Underwater acoustics',
    path: '/acoustics',
    summary: 'Turn ship sounds into spectrograms, then see how FedAvg weights clients with different amounts of data.',
    topics: ['Spectrograms', 'EfficientNet-B0', 'FedAvg'],
    owner: 'Teammate B',
    accent: 'var(--accent-acoustics)',
    Page: lazy(() => import('../modules/acoustics/AcousticsPage')),
  },
  {
    id: 'hardware',
    title: 'Hardware-aware learning',
    shortTitle: 'Hardware-aware',
    path: '/hardware',
    summary: 'Estimate the time and energy a device spends on one federated round, and test it against a budget.',
    topics: ['Device profiles', 'Bandwidth', 'Energy budget'],
    owner: 'Teammate C',
    accent: 'var(--accent-hardware)',
    Page: lazy(() => import('../modules/hardware/HardwarePage')),
  },
];
