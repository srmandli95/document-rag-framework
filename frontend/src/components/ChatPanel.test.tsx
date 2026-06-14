import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ChatPanel } from "./ChatPanel";

const baseProps = {
  messages: [],
  readyDocumentCount: 1,
  loading: false,
  error: null,
  onSend: vi.fn().mockResolvedValue(undefined),
};

describe("ChatPanel", () => {
  it("submits a question once", async () => {
    const onSend = vi.fn().mockResolvedValue(undefined);
    render(<ChatPanel {...baseProps} onSend={onSend} />);
    await userEvent.type(screen.getByLabelText("Ask a question"), "What is covered?");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onSend).toHaveBeenCalledWith("What is covered?");
  });

  it("disables chat until a document is ready", () => {
    render(<ChatPanel {...baseProps} readyDocumentCount={0} />);
    expect(screen.getByLabelText("Ask a question")).toBeDisabled();
    expect(screen.getByText("Add a ready document to begin")).toBeInTheDocument();
  });

  it("renders API errors", () => {
    render(<ChatPanel {...baseProps} error="Answer failed" />);
    expect(screen.getByRole("alert")).toHaveTextContent("Answer failed");
  });

  it("does not render the old session dropdown", () => {
    render(<ChatPanel {...baseProps} />);
    expect(screen.queryByLabelText("Chat session")).not.toBeInTheDocument();
  });

  it("shows each source document once and hides chunk metadata from legacy answers", () => {
    render(
      <ChatPanel
        {...baseProps}
        messages={[
          {
            question: "What is the late fee?",
            answer: "The late fee is 2%.\n\nSource: guide.docx, Chunk ID: chunk-1",
            citations: [
              { chunk_id: "chunk-1", document_id: "doc-1", document_name: "guide.docx" },
              { chunk_id: "chunk-2", document_id: "doc-1", document_name: "guide.docx" },
            ],
          },
        ]}
      />,
    );

    expect(screen.getByText("The late fee is 2%.")).toBeInTheDocument();
    expect(screen.queryByText(/Chunk ID/)).not.toBeInTheDocument();
    expect(screen.getAllByText("guide.docx")).toHaveLength(1);
  });
});
