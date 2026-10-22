import { createBrowserRouter, Link } from 'react-router-dom';
import { Card } from '../components/Card';
import { HomePage } from './HomePage';
import { Layout } from './Layout';
import { MODULES } from './modules';

function NotFound() {
  return (
    <div className="page">
      <Card title="Page not found" headingLevel={2}>
        <p>This page does not exist.</p>
        <div>
          <Link to="/" className="btn btn-primary">
            Back to overview
          </Link>
        </div>
      </Card>
    </div>
  );
}

export const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <HomePage /> },
      ...MODULES.map(({ path, Page }) => ({ path, element: <Page /> })),
      { path: '*', element: <NotFound /> },
    ],
  },
]);
