import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { AuthGate } from "./AuthGate";

vi.mock("../api/client", () => ({
  api: {
    me: vi.fn(),
    logout: vi.fn(),
    googleLogin: vi.fn(),
  },
}));

const user = {
  id: "user-1",
  email: "srikar@example.com",
  full_name: "Srikar Mandli",
  auth_provider: "google",
};

describe("AuthGate", () => {
  beforeEach(() => {
    vi.mocked(api.me).mockReset();
    vi.mocked(api.logout).mockReset();
    vi.mocked(api.googleLogin).mockReset();
  });

  it("shows the app after loading the current user", async () => {
    vi.mocked(api.me).mockResolvedValue(user);

    render(<AuthGate><div>Workspace</div></AuthGate>);

    expect(await screen.findByText("Workspace")).toBeInTheDocument();
    expect(screen.getByText("Signed in as")).toBeInTheDocument();
    expect(screen.getByText("Srikar Mandli")).toBeInTheDocument();
  });

  it("rechecks auth when the browser returns focus after login", async () => {
    vi.mocked(api.me)
      .mockRejectedValueOnce(new Error("Not authenticated"))
      .mockResolvedValueOnce(user);

    render(<AuthGate><div>Workspace</div></AuthGate>);

    expect(await screen.findByRole("button", { name: "Continue with Google" })).toBeInTheDocument();
    fireEvent.focus(window);

    expect(await screen.findByText("Workspace")).toBeInTheDocument();
    expect(vi.mocked(api.me)).toHaveBeenCalledTimes(2);
  });

  it("disables the Google button while starting login", async () => {
    vi.mocked(api.me).mockRejectedValue(new Error("Not authenticated"));
    vi.mocked(api.googleLogin).mockResolvedValue(undefined);

    render(<AuthGate><div>Workspace</div></AuthGate>);

    const loginButton = await screen.findByRole("button", { name: "Continue with Google" });
    fireEvent.click(loginButton);

    await waitFor(() => expect(api.googleLogin).toHaveBeenCalledOnce());
    expect(screen.getByRole("button", { name: "Opening Google..." })).toBeDisabled();
  });
});
