import { type FormEvent, type ReactNode, useEffect, useState } from "react";
import { api } from "../api/client";
import type { AuthUser, Organization } from "../types";

export function AuthGate({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>();
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [activeOrganizationId, setActiveOrganizationId] = useState<string>();
  const [registering, setRegistering] = useState(false);
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string>();

  const loadOrganizations = async () => {
    const data = await api.listOrganizations();
    setOrganizations(data.organizations);
    setActiveOrganizationId(data.active_organization_id || undefined);
  };

  useEffect(() => {
    api.me()
      .then(async (currentUser) => {
        setUser(currentUser);
        await loadOrganizations();
      })
      .catch(() => setUser(null));
  }, []);

  const submitCredentials = async (event: FormEvent) => {
    event.preventDefault();
    setError(undefined);
    try {
      const authenticatedUser = registering
        ? await api.register(email, password, fullName)
        : await api.login(email, password);
      setUser(authenticatedUser);
      await loadOrganizations();
    } catch (authError) {
      setError(authError instanceof Error ? authError.message : "Authentication failed");
    }
  };

  const createOrganization = async () => {
    const name = window.prompt("Organization name");
    if (!name?.trim()) return;
    const organization = await api.createOrganization(name);
    await api.selectOrganization(organization.id);
    window.location.reload();
  };

  if (user === undefined) return <div className="auth-page">Loading...</div>;

  if (!user) {
    return (
      <main className="auth-page">
        <form className="auth-card" onSubmit={submitCredentials}>
          <p className="eyebrow">Document RAG</p>
          <h1>{registering ? "Create local account" : "Sign in"}</h1>
          <p className="muted">
            {registering
              ? "Create an account to test organizations and shared knowledge locally."
              : "Use your organization account to access its knowledge base."}
          </p>
          {error && <div className="error-banner">{error}</div>}
          {registering && (
            <label>Full name<input value={fullName} onChange={(event) => setFullName(event.target.value)} /></label>
          )}
          <label>Email<input type="email" value={email} onChange={(event) => setEmail(event.target.value)} required /></label>
          <label>Password<input type="password" minLength={6} value={password} onChange={(event) => setPassword(event.target.value)} required /></label>
          <button className="primary-button" type="submit">{registering ? "Create account" : "Sign in"}</button>
          {!registering && <button className="secondary-button" type="button" onClick={() => void api.googleLogin()}>Continue with Google</button>}
          <button
            className="text-button"
            type="button"
            onClick={() => {
              setRegistering((current) => !current);
              setError(undefined);
            }}
          >
            {registering ? "Already have an account? Sign in" : "Need a local account? Create one"}
          </button>
        </form>
      </main>
    );
  }

  return (
    <div className="authenticated-shell">
      <header className="account-bar">
        <strong>{user.full_name || user.email}</strong>
        {user.auth_provider !== "dev" && (
          <>
            <select
              aria-label="Active organization"
              value={activeOrganizationId || ""}
              onChange={async (event) => {
                await api.selectOrganization(event.target.value || null);
                window.location.reload();
              }}
            >
              <option value="">Private workspace</option>
              {organizations.map((organization) => (
                <option key={organization.id} value={organization.id}>{organization.name} ({organization.role})</option>
              ))}
            </select>
            <button className="secondary-button" onClick={() => void createOrganization()}>New organization</button>
            <button className="secondary-button" onClick={async () => { await api.logout(); setUser(null); }}>Sign out</button>
          </>
        )}
      </header>
      {children}
    </div>
  );
}
