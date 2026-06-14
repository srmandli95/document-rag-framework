import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api/client";
import { ChatPanel } from "./components/ChatPanel";
import { DocumentPanel } from "./components/DocumentPanel";
import type { ChatMessage, ChatSession, DocumentRecord } from "./types";

const activeStatuses = new Set(["uploading", "validating", "extracting", "chunking", "embedding"]);

export default function App() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [documentError, setDocumentError] = useState<string | null>(null);
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>();
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);

  const loadDocuments = useCallback(async () => {
    try {
      setDocuments(await api.listDocuments());
      setDocumentError(null);
    } catch (error) {
      setDocumentError(error instanceof Error ? error.message : "Could not load documents");
    } finally {
      setDocumentsLoading(false);
    }
  }, []);

  const loadSessions = useCallback(async () => {
    try {
      setSessions(await api.listSessions());
    } catch (error) {
      setChatError(error instanceof Error ? error.message : "Could not load chat sessions");
    }
  }, []);

  useEffect(() => {
    void loadDocuments();
    void loadSessions();
  }, [loadDocuments, loadSessions]);

  const processing = documents.some((document) => activeStatuses.has(document.display_status));
  useEffect(() => {
    if (!processing) return;
    const interval = window.setInterval(() => void loadDocuments(), 2000);
    return () => window.clearInterval(interval);
  }, [processing, loadDocuments]);

  const readyDocumentCount = useMemo(
    () => documents.filter((document) => document.display_status === "ready").length,
    [documents],
  );

  const upload = async (file: File) => {
    setDocumentError(null);
    try {
      const uploaded = await api.uploadDocument(file);
      setDocuments((current) => [uploaded, ...current.filter((item) => item.document_id !== uploaded.document_id)]);
      await loadDocuments();
    } catch (error) {
      setDocumentError(error instanceof Error ? error.message : "Upload failed");
    }
  };

  const remove = async (document: DocumentRecord) => {
    setDocumentError(null);
    try {
      await api.deleteDocument(document.document_id);
      setDocuments((current) => current.filter((item) => item.document_id !== document.document_id));
    } catch (error) {
      setDocumentError(error instanceof Error ? error.message : "Delete failed");
    }
  };

  const selectSession = async (sessionId: string) => {
    setChatError(null);
    try {
      setMessages(await api.getSession(sessionId));
      setActiveSessionId(sessionId);
    } catch (error) {
      setChatError(error instanceof Error ? error.message : "Could not load chat");
    }
  };

  const send = async (question: string) => {
    if (chatLoading) return;
    setChatLoading(true);
    setChatError(null);
    setMessages((current) => [...current, { question, citations: [] }]);
    try {
      const response = await api.ask(question, activeSessionId);
      setMessages((current) => [...current.slice(0, -1), response]);
      setActiveSessionId(response.session_id);
      await loadSessions();
    } catch (error) {
      setMessages((current) => current.slice(0, -1));
      setChatError(error instanceof Error ? error.message : "Could not generate an answer");
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <DocumentPanel
        documents={documents}
        loading={documentsLoading}
        error={documentError}
        onUpload={upload}
        onDelete={remove}
      />
      <ChatPanel
        messages={messages}
        sessions={sessions}
        activeSessionId={activeSessionId}
        readyDocumentCount={readyDocumentCount}
        loading={chatLoading}
        error={chatError}
        onSend={send}
        onSelectSession={selectSession}
        onNewChat={() => { setActiveSessionId(undefined); setMessages([]); setChatError(null); }}
      />
    </div>
  );
}
