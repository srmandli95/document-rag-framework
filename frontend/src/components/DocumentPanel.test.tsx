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

describe("DocumentPanel", () => {
  it("uploads a selected supported file", async () => {
    const onUpload = vi.fn().mockResolvedValue(undefined);
    render(<DocumentPanel documents={[]} loading={false} error={null} onUpload={onUpload} onDelete={vi.fn()} />);
    const file = new File(["hello"], "guide.txt", { type: "text/plain" });
    await userEvent.upload(screen.getByLabelText("Choose document"), file);
    expect(onUpload).toHaveBeenCalledWith(file);
  });

  it("shows processing status and failure errors", () => {
    render(<DocumentPanel documents={[document]} loading={false} error="Upload failed" onUpload={vi.fn()} onDelete={vi.fn()} />);
    expect(screen.getByText("embedding")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("Upload failed");
  });

  it("confirms and removes a document", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const onDelete = vi.fn().mockResolvedValue(undefined);
    render(<DocumentPanel documents={[document]} loading={false} error={null} onUpload={vi.fn()} onDelete={onDelete} />);
    fireEvent.click(screen.getByLabelText("Delete guide.pdf"));
    expect(onDelete).toHaveBeenCalledWith(document);
  });
});
