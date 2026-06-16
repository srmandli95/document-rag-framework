import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api/client";
import { ChatPanel } from "./components/ChatPanel";
import { ChatHistoryPanel } from "./components/ChatHistoryPanel";
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
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [documentsCollapsed, setDocumentsCollapsed] = useState(false);
  const [historyCollapsed, setHistoryCollapsed] = useState(false);

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
      setHistoryError(null);
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : "Could not load chat sessions");
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

  const newChat = () => {
    setActiveSessionId(undefined);
    setMessages([]);
    setChatError(null);
  };

  const deleteSession = async (session: ChatSession) => {
    setHistoryError(null);
    try {
      await api.deleteSession(session.session_id);
      setSessions((current) => current.filter((item) => item.session_id !== session.session_id));
      if (activeSessionId === session.session_id) {
        newChat();
      }
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : "Could not delete chat");
    }
  };

  return (
    <div className={`app-shell ${documentsCollapsed ? "documents-collapsed" : ""} ${historyCollapsed ? "history-collapsed" : ""}`}>
      <DocumentPanel
        documents={documents}
        loading={documentsLoading}
        error={documentError}
        collapsed={documentsCollapsed}
        onToggleCollapsed={() => setDocumentsCollapsed((current) => !current)}
        onUpload={upload}
        onDelete={remove}
      />
      <ChatHistoryPanel
        sessions={sessions}
        activeSessionId={activeSessionId}
        error={historyError}
        collapsed={historyCollapsed}
        onToggleCollapsed={() => setHistoryCollapsed((current) => !current)}
        onSelect={selectSession}
        onDelete={deleteSession}
        onNewChat={newChat}
      />
      <ChatPanel
        messages={messages}
        readyDocumentCount={readyDocumentCount}
        loading={chatLoading}
        error={chatError}
        onSend={send}
      />
    </div>
  );
}
