import { type ReactNode, useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { AuthUser } from "../types";

export function AuthGate({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>();
  const [error, setError] = useState<string>();
  const [loginLoading, setLoginLoading] = useState(false);

  const refreshUser = useCallback(async () => {
    try {
      setUser(await api.me());
    } catch {
      setUser(null);
    }
  }, []);

  useEffect(() => {
    void refreshUser();
  }, [refreshUser]);

  useEffect(() => {
    const refreshWhenVisible = () => {
      if (document.visibilityState === "visible") {
        void refreshUser();
      }
    };
    const refreshFromPageCache = () => void refreshUser();

    window.addEventListener("focus", refreshWhenVisible);
    window.addEventListener("pageshow", refreshFromPageCache);
    document.addEventListener("visibilitychange", refreshWhenVisible);

    return () => {
      window.removeEventListener("focus", refreshWhenVisible);
      window.removeEventListener("pageshow", refreshFromPageCache);
      document.removeEventListener("visibilitychange", refreshWhenVisible);
    };
  }, [refreshUser]);

  const googleLogin = async () => {
    setError(undefined);
    setLoginLoading(true);
    try {
      await api.googleLogin();
    } catch (authError) {
      setError(authError instanceof Error ? authError.message : "Google sign-in failed");
      setLoginLoading(false);
    }
  };

  if (user === undefined) return <main className="auth-page">Loading...</main>;

  if (!user) {
    return (
      <main className="auth-page">
        <section className="oauth-card">
          <p className="eyebrow">Document RAG</p>
          <h1>Your document workspace</h1>
          <p className="muted">Continue with Google, then upload documents and start asking questions.</p>
          {error && <div className="error-banner">{error}</div>}
          <button className="google-button" type="button" disabled={loginLoading} onClick={() => void googleLogin()}>
            {loginLoading ? "Opening Google..." : "Continue with Google"}
          </button>
        </section>
      </main>
    );
  }

  return (
    <div className="authenticated-shell">
      <header className="account-bar">
        <span>Signed in as <strong>{user.full_name || user.email}</strong></span>
        <button className="secondary-button" onClick={async () => { await api.logout(); setUser(null); }}>
          Logout
        </button>
      </header>
      {children}
    </div>
  );
}
