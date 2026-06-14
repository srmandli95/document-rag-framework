import type { ChatSession } from "../types";

interface Props {
  sessions: ChatSession[];
  activeSessionId?: string;
  error: string | null;
  onSelect: (sessionId: string) => Promise<void>;
  onDelete: (session: ChatSession) => Promise<void>;
  onNewChat: () => void;
}

export function ChatHistoryPanel({
  sessions,
  activeSessionId,
  error,
  onSelect,
  onDelete,
  onNewChat,
}: Props) {
  const confirmDelete = async (session: ChatSession) => {
    if (window.confirm(`Delete chat "${session.title}"?`)) {
      await onDelete(session);
    }
  };

  return (
    <aside className="history-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Conversations</p>
          <h2>Chat history</h2>
        </div>
        <span className="count">{sessions.length}</span>
      </div>

      <button type="button" className="primary-button new-chat-button" onClick={onNewChat}>
        + New chat
      </button>

      {error && <div className="error-banner" role="alert">{error}</div>}

      <div className="history-list">
        {sessions.length === 0 && (
          <p className="muted empty-history">Your previous chats will appear here.</p>
        )}
        {sessions.map((session) => (
          <article
            className={`history-item ${activeSessionId === session.session_id ? "active" : ""}`}
            key={session.session_id}
          >
            <button
              type="button"
              className="history-select"
              aria-label={`Open ${session.title}`}
              onClick={() => void onSelect(session.session_id)}
            >
              <strong>{session.title}</strong>
              {session.updated_at && <span>{new Date(session.updated_at).toLocaleDateString()}</span>}
            </button>
            <button
              type="button"
              className="icon-button history-delete"
              aria-label={`Delete chat ${session.title}`}
              onClick={() => void confirmDelete(session)}
            >
              ×
            </button>
          </article>
        ))}
      </div>
    </aside>
  );
}
