import { FormEvent, useState } from "react";
import type { ChatMessage, ChatSession } from "../types";

interface Props {
  messages: ChatMessage[];
  sessions: ChatSession[];
  activeSessionId?: string;
  readyDocumentCount: number;
  loading: boolean;
  error: string | null;
  onSend: (question: string) => Promise<void>;
  onSelectSession: (sessionId: string) => Promise<void>;
  onNewChat: () => void;
}

export function ChatPanel(props: Props) {
  const [question, setQuestion] = useState("");
  const disabled = props.loading || props.readyDocumentCount === 0;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const cleanQuestion = question.trim();
    if (!cleanQuestion || disabled) return;
    setQuestion("");
    await props.onSend(cleanQuestion);
  };

  return (
    <main className="chat-panel">
      <header className="chat-header">
        <div>
          <p className="eyebrow">Document RAG</p>
          <h2>Chat with your sources</h2>
        </div>
        <div className="session-controls">
          <select
            aria-label="Chat session"
            value={props.activeSessionId || ""}
            onChange={(event) => event.target.value && void props.onSelectSession(event.target.value)}
          >
            <option value="">New chat</option>
            {props.sessions.map((session) => (
              <option key={session.session_id} value={session.session_id}>{session.title}</option>
            ))}
          </select>
          <button type="button" className="secondary-button" onClick={props.onNewChat}>New chat</button>
        </div>
      </header>

      <section className="messages" aria-live="polite">
        {props.messages.length === 0 && (
          <div className="chat-empty">
            <h3>{props.readyDocumentCount ? "Ask a grounded question" : "Add a ready document to begin"}</h3>
            <p>
              {props.readyDocumentCount
                ? "Answers will include citations from your knowledge base."
                : "Documents become available here after extraction, chunking, and embedding finish."}
            </p>
          </div>
        )}
        {props.messages.map((message, index) => (
          <div className="message-pair" key={message.message_id || index}>
            <div className="message user-message">{message.question}</div>
            {message.answer && (
              <div className="message assistant-message">
                <p>{message.answer}</p>
                {message.citations.length > 0 && (
                  <div className="citations">
                    <strong>Sources</strong>
                    {message.citations.map((citation, citationIndex) => (
                      <span key={`${citation.chunk_id || "source"}-${citationIndex}`}>
                        {citation.document_name || citation.source || `Source ${citationIndex + 1}`}
                        {citation.page_number ? `, page ${citation.page_number}` : ""}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
        {props.loading && <div className="message assistant-message loading-message">Generating answer…</div>}
      </section>

      {props.error && <div className="error-banner chat-error" role="alert">{props.error}</div>}
      <form className="chat-form" onSubmit={submit}>
        <textarea
          aria-label="Ask a question"
          placeholder={props.readyDocumentCount ? "Ask about your documents…" : "Waiting for a ready document…"}
          value={question}
          disabled={disabled}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }}
        />
        <button type="submit" className="primary-button" disabled={disabled || !question.trim()}>
          Send
        </button>
      </form>
    </main>
  );
}
