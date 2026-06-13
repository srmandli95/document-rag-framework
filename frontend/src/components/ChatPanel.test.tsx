import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ChatPanel } from "./ChatPanel";

const baseProps = {
  messages: [],
  sessions: [],
  readyDocumentCount: 1,
  loading: false,
  error: null,
  onSend: vi.fn().mockResolvedValue(undefined),
  onSelectSession: vi.fn().mockResolvedValue(undefined),
  onNewChat: vi.fn(),
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
});
