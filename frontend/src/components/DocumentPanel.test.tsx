import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { DocumentPanel } from "./DocumentPanel";
import type { DocumentRecord } from "../types";

const document: DocumentRecord = {
  document_id: "doc-1",
  original_file_name: "guide.pdf",
  file_size_bytes: 2048,
  status: "processing",
  display_status: "embedding",
  created_at: "2026-06-13T12:00:00Z",
};

const baseProps = {
  documents: [],
  loading: false,
  error: null,
  collapsed: false,
  onToggleCollapsed: vi.fn(),
  onUpload: vi.fn().mockResolvedValue(undefined),
  onDelete: vi.fn().mockResolvedValue(undefined),
};

describe("DocumentPanel", () => {
  it("uploads a selected supported file", async () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    render(<DocumentPanel {...baseProps} onUpload={onUpload} />);
    const file = new File(["hello"], "guide.txt", { type: "text/plain" });
    await userEvent.upload(screen.getByLabelText("Choose document"), file);
    expect(onUpload).toHaveBeenCalledWith(file);
  });

  it("shows processing status and failure errors", () => {
    render(<DocumentPanel {...baseProps} documents={[document]} error="Upload failed" />);
    expect(screen.getByText("embedding")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("Upload failed");
  });

  it("confirms and removes a document", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const onDelete = vi.fn().mockResolvedValue(undefined);
    render(<DocumentPanel {...baseProps} documents={[document]} onDelete={onDelete} />);
    fireEvent.click(screen.getByLabelText("Delete guide.pdf"));
    expect(onDelete).toHaveBeenCalledWith(document);
  });

  it("collapses and expands the documents panel", () => {
    const onToggleCollapsed = vi.fn();
    const { rerender } = render(<DocumentPanel {...baseProps} onToggleCollapsed={onToggleCollapsed} />);
    fireEvent.click(screen.getByLabelText("Collapse documents"));
    expect(onToggleCollapsed).toHaveBeenCalledOnce();

    rerender(<DocumentPanel {...baseProps} collapsed onToggleCollapsed={onToggleCollapsed} />);
    fireEvent.click(screen.getByLabelText("Expand documents"));
    expect(onToggleCollapsed).toHaveBeenCalledTimes(2);
  });
});
