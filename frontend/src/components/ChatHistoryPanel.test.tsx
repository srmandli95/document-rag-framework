import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ChatHistoryPanel } from "./ChatHistoryPanel";

const session = {
  session_id: "session-1",
  title: "What is covered?",
  updated_at: "2026-06-14T10:00:00Z",
};

const baseProps = {
  sessions: [session],
  activeSessionId: "session-1",
  error: null,
  collapsed: false,
  onToggleCollapsed: vi.fn(),
  onSelect: vi.fn().mockResolvedValue(undefined),
  onDelete: vi.fn().mockResolvedValue(undefined),
  onNewChat: vi.fn(),
};

describe("ChatHistoryPanel", () => {
  it("opens a previous chat from the history panel", () => {
    const onSelect = vi.fn().mockResolvedValue(undefined);
    render(<ChatHistoryPanel {...baseProps} onSelect={onSelect} />);
    fireEvent.click(screen.getByLabelText("Open What is covered?"));
    expect(onSelect).toHaveBeenCalledWith("session-1");
  });

  it("confirms before deleting chat history", () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const onDelete = vi.fn().mockResolvedValue(undefined);
    render(<ChatHistoryPanel {...baseProps} onDelete={onDelete} />);
    fireEvent.click(screen.getByLabelText("Delete chat What is covered?"));
    expect(onDelete).toHaveBeenCalledWith(session);
  });

  it("starts a new chat", () => {
    const onNewChat = vi.fn();
    render(<ChatHistoryPanel {...baseProps} onNewChat={onNewChat} />);
    fireEvent.click(screen.getByRole("button", { name: "+ New chat" }));
    expect(onNewChat).toHaveBeenCalledOnce();
  });

  it("collapses and expands chat history", () => {
    const onToggleCollapsed = vi.fn();
    const { rerender } = render(<ChatHistoryPanel {...baseProps} onToggleCollapsed={onToggleCollapsed} />);
    fireEvent.click(screen.getByLabelText("Collapse chat history"));
    expect(onToggleCollapsed).toHaveBeenCalledOnce();

    rerender(<ChatHistoryPanel {...baseProps} collapsed onToggleCollapsed={onToggleCollapsed} />);
    fireEvent.click(screen.getByLabelText("Expand chat history"));
    expect(onToggleCollapsed).toHaveBeenCalledTimes(2);
  });
});
