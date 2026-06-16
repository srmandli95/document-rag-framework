import { useRef, useState } from "react";
import type { DocumentRecord } from "../types";

interface Props {
  documents: DocumentRecord[];
  loading: boolean;
  error: string | null;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onUpload: (file: File) => Promise<void>;
  onDelete: (document: DocumentRecord) => Promise<void>;
}

const allowedExtensions = ".pdf,.docx,.txt,.md,.html";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function DocumentPanel({
  documents,
  loading,
  error,
  collapsed,
  onToggleCollapsed,
  onUpload,
  onDelete,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const uploadFirst = async (files: FileList | null) => {
    const file = files?.[0];
    if (file) await onUpload(file);
  };

  const confirmDelete = async (document: DocumentRecord) => {
    if (window.confirm(`Delete "${document.original_file_name}" and all generated data?`)) {
      await onDelete(document);
    }
  };

  return (
    <aside className={`document-panel collapsible-panel ${collapsed ? "collapsed" : ""}`} aria-label="Documents">
      {!collapsed && (
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Knowledge base</p>
            <h1>Documents</h1>
          </div>
          <div className="panel-actions">
            <span className="count">{documents.length}</span>
            <button
              type="button"
              className="icon-button collapse-button"
              aria-label="Collapse documents"
              aria-expanded="true"
              onClick={onToggleCollapsed}
            >
              ‹
            </button>
          </div>
        </div>
      )}

      {collapsed && (
        <button
          type="button"
          className="collapsed-panel-button"
          aria-label="Expand documents"
          onClick={onToggleCollapsed}
        >
          <span className="collapsed-panel-title">Documents</span>
          <span className="count">{documents.length}</span>
        </button>
      )}

      {!collapsed && (
        <>
          <div
            className={`drop-zone ${dragging ? "dragging" : ""}`}
            onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              void uploadFirst(event.dataTransfer.files);
            }}
          >
            <strong>Drop a document here</strong>
            <span>PDF, DOCX, TXT, Markdown, or HTML</span>
            <button type="button" className="secondary-button" onClick={() => inputRef.current?.click()}>
              Choose file
            </button>
            <input
              ref={inputRef}
              aria-label="Choose document"
              type="file"
              accept={allowedExtensions}
              hidden
              onChange={(event) => void uploadFirst(event.target.files)}
            />
          </div>

          {error && <div className="error-banner" role="alert">{error}</div>}

          <div className="document-list" aria-busy={loading}>
            {!loading && documents.length === 0 && (
              <p className="muted empty-documents">Your uploaded documents will appear here.</p>
            )}
            {documents.map((document) => (
              <article className="document-card" key={document.document_id}>
                <div className="document-card-top">
                  <strong title={document.original_file_name}>{document.original_file_name}</strong>
                  <button
                    type="button"
                    className="icon-button"
                    aria-label={`Delete ${document.original_file_name}`}
                    onClick={() => void confirmDelete(document)}
                  >
                    ×
                  </button>
                </div>
                <div className="document-meta">
                  <span>{formatBytes(document.file_size_bytes)}</span>
                  <span>{document.created_at ? new Date(document.created_at).toLocaleDateString() : "Just now"}</span>
                </div>
                <span className={`status status-${document.display_status}`}>{document.display_status}</span>
                {document.failure_reason && <p className="failure-reason">{document.failure_reason}</p>}
              </article>
            ))}
          </div>
        </>
      )}
    </aside>
  );
}
