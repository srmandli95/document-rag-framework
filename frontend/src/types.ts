export type DisplayStatus =
  | "uploading"
  | "validating"
  | "extracting"
  | "chunking"
  | "embedding"
  | "ready"
  | "failed";

export interface DocumentRecord {
  document_id: string;
  original_file_name: string;
  file_size_bytes: number;
  status: string;
  display_status: DisplayStatus;
  failure_reason?: string | null;
  created_at?: string | null;
}

export interface Citation {
  document_id?: string;
  document_name?: string;
  source?: string;
  page_number?: number | null;
  chunk_id?: string;
  [key: string]: unknown;
}

export interface ChatMessage {
  message_id?: string;
  question: string;
  answer?: string | null;
  citations: Citation[];
}

export interface ChatSession {
  session_id: string;
  title: string;
  updated_at?: string;
}

export interface AuthUser {
  id: string;
  email: string;
  full_name?: string | null;
  auth_provider: string;
}
