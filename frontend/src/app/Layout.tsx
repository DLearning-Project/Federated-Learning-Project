import { Suspense, useEffect, useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { BrandMark, CloseIcon, MenuIcon } from '../components/Icons';
import { LoadingState } from '../components/StatusMessage';
import { BackendStatus } from './BackendStatus';
import { MODULES } from './modules';

export function Layout() {
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();

  // Close the mobile menu and return to the top whenever the route changes.
  useEffect(() => {
    setMenuOpen(false);
    window.scrollTo(0, 0);
  }, [location.pathname]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <NavLink to="/" className="brand">
          <BrandMark className="brand-mark" />
          FL Explorer
        </NavLink>
        <button type="button" className="btn btn-ghost" aria-expanded={menuOpen} aria-controls="sidebar" onClick={() => setMenuOpen(true)}>
          <MenuIcon /> Menu
        </button>
      </header>

      <nav id="sidebar" className={`sidebar ${menuOpen ? 'open' : ''}`} aria-label="Main navigation">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 'var(--space-2)' }}>
          <NavLink to="/" className="brand">
            <BrandMark className="brand-mark" />
            <span>
              Federated Learning
              <br />
              Explorer
            </span>
          </NavLink>
          {menuOpen && (
            <button type="button" className="btn btn-ghost" onClick={() => setMenuOpen(false)} aria-label="Close menu">
              <CloseIcon />
            </button>
          )}
        </div>

        <ul className="nav-list">
          <li>
            <NavLink to="/" end className="nav-link">
              <span className="nav-dot" style={{ borderStyle: 'dashed' }} aria-hidden />
              Overview
            </NavLink>
          </li>
        </ul>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
          <span className="eyebrow">Modules</span>
          <ul className="nav-list">
            {MODULES.map((m) => (
              <li key={m.id}>
                <NavLink to={m.path} className="nav-link">
                  <span className="nav-dot" style={{ background: m.accent }} aria-hidden />
                  {m.shortTitle}
                </NavLink>
              </li>
            ))}
          </ul>
        </div>

        <div className="sidebar-footer">
          <BackendStatus />
          <p>Results are temporary and reset when you refresh the page or restart the server.</p>
        </div>
      </nav>

      <main className="main" id="main">
        <Suspense fallback={<LoadingState label="Loading module…" />}>
          <Outlet />
        </Suspense>
      </main>
    </div>
  );
}
