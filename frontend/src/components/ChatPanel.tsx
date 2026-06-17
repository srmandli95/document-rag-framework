import { FormEvent, useState } from "react";
import type { ChatMessage } from "../types";

function cleanAnswer(answer: string): string {
  return answer
    .split("\n")
    .filter((line) => {
      const cleanLine = line.trim();
      return !/^(sources?|citations?)\s*:?\s*$/i.test(cleanLine)
        && !/\bchunk\s+id\s*:/i.test(cleanLine);
    })
    .join("\n")
    .trim();
}

function uniqueSources(citations: ChatMessage["citations"]) {
  const sources = new Map<string, ChatMessage["citations"][number]>();

  citations.forEach((citation, index) => {
    const label = citation.document_name || citation.source || `Source ${index + 1}`;
    const key = String(citation.document_id || label);
    if (!sources.has(key)) {
      sources.set(key, citation);
    }
  });

  return [...sources.values()];
}

interface Props {
  messages: ChatMessage[];
  readyDocumentCount: number;
  loading: boolean;
  error: string | null;
  onSend: (question: string) => Promise<void>;
}

export function ChatPanel(props: Props) {
  const [question, setQuestion] = useState("");
  const disabled = props.readyDocumentCount === 0;
  const hasPendingMessages = props.messages.some((message) => message.is_pending);

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
                <p>{cleanAnswer(message.answer)}</p>
                {uniqueSources(message.citations).length > 0 && (
                  <div className="citations">
                    <strong>Sources</strong>
                    {uniqueSources(message.citations).map((citation, citationIndex) => (
                      <span key={String(citation.document_id || citation.document_name || citation.source || citationIndex)}>
                        {citation.document_name || citation.source || `Source ${citationIndex + 1}`}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
            {message.is_pending && (
              <div className="message assistant-message loading-message">Generating answer…</div>
            )}
          </div>
        ))}
        {props.loading && !hasPendingMessages && (
          <div className="message assistant-message loading-message">Generating answer…</div>
        )}
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
