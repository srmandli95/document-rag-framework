import type { AuthUser, ChatMessage, ChatSession, Citation, DocumentRecord, Organization } from "../types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

function authHeaders(): HeadersInit {
  const token = localStorage.getItem("rag_access_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    credentials: "include",
    headers: {
      ...authHeaders(),
      ...options.headers,
    },
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail || `Request failed (${response.status})`);
  }

  return response.json() as Promise<T>;
}

export const api = {
  async me(): Promise<AuthUser> {
    return request<AuthUser>("/auth/me");
  },

  async login(email: string, password: string): Promise<AuthUser> {
    const data = await request<{ user: AuthUser }>("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    localStorage.removeItem("rag_access_token");
    return data.user;
  },

  async register(email: string, password: string, fullName: string): Promise<AuthUser> {
    const data = await request<{ user: AuthUser }>("/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, full_name: fullName || null }),
    });
    localStorage.removeItem("rag_access_token");
    return data.user;
  },

  async logout(): Promise<void> {
    await request("/auth/logout", { method: "POST" });
    localStorage.removeItem("rag_access_token");
  },

  async googleLogin(): Promise<void> {
    const data = await request<{ authorization_url: string }>("/auth/google/login");
    window.location.assign(data.authorization_url);
  },

  async listOrganizations(): Promise<{ organizations: Organization[]; active_organization_id?: string | null }> {
    return request("/auth/organizations");
  },

  async createOrganization(name: string): Promise<Organization> {
    return request("/auth/organizations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
  },

  async selectOrganization(organizationId: string | null): Promise<void> {
    await request("/auth/organizations/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ organization_id: organizationId }),
    });
  },

  async listDocuments(): Promise<DocumentRecord[]> {
    const data = await request<{ documents: DocumentRecord[] }>("/documents");
    return data.documents;
  },

  async uploadDocument(file: File): Promise<DocumentRecord> {
    const body = new FormData();
    body.append("file", file);
    body.append("category", "general");
    return request<DocumentRecord>("/documents/upload", { method: "POST", body });
  },

  async deleteDocument(documentId: string): Promise<void> {
    await request(`/documents/${documentId}`, { method: "DELETE" });
  },

  async listSessions(): Promise<ChatSession[]> {
    const data = await request<{ sessions: ChatSession[] }>("/chat/sessions");
    return data.sessions;
  },

  async getSession(sessionId: string): Promise<ChatMessage[]> {
    const data = await request<{ messages: ChatMessage[] }>(`/chat/sessions/${sessionId}`);
    return data.messages;
  },

  async deleteSession(sessionId: string): Promise<void> {
    await request(`/chat/sessions/${sessionId}`, { method: "DELETE" });
  },

  async ask(question: string, sessionId?: string): Promise<ChatMessage & { session_id: string }> {
    const data = await request<{
      message_id: string;
      session_id: string;
      question: string;
      answer: string;
      citations: Citation[];
    }>("/chat/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId }),
    });
    return data;
  },
};
