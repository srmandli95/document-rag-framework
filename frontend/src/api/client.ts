import type { ChatMessage, ChatSession, Citation, DocumentRecord } from "../types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

function authHeaders(): HeadersInit {
  const token = localStorage.getItem("rag_access_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
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
