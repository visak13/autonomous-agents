import { useTheme } from "../../theme/useTheme";
import type { StreamStatus } from "../../hooks/useChatStream";
import "./TopBar.css";

export type AppView = "tasks" | "spec" | "tuned" | "shapes";

interface TopBarProps {
  streamStatus: StreamStatus;
  hasChat: boolean;
  view: AppView;
  onSetView: (view: AppView) => void;
}

const STREAM_LABEL: Record<StreamStatus, string> = {
  idle: "No stream",
  connecting: "Connecting…",
  open: "Live",
  reconnecting: "Reconnecting…",
};

export function TopBar({ streamStatus, hasChat, view, onSetView }: TopBarProps) {
  const { theme, toggle } = useTheme();
  // Each entry toggles its view: clicking it when already open returns to tasks.
  const toggleTo = (target: AppView) => () =>
    onSetView(view === target ? "tasks" : target);
  return (
    <header className="topbar">
      <div className="topbar-brand">
        <span className="topbar-logo" aria-hidden="true">
          λ
        </span>
        <span className="topbar-title">ReactiveAgents</span>
      </div>

      <div className="topbar-right">
        <button
          type="button"
          className={`topbar-view-toggle${view === "tuned" ? " topbar-view-active" : ""}`}
          onClick={toggleTo("tuned")}
          aria-pressed={view === "tuned"}
          title="How your model was tuned for your PC"
        >
          {view === "tuned" ? "← Task chat" : "🔧 How it's tuned"}
        </button>
        <button
          type="button"
          className={`topbar-view-toggle${view === "spec" ? " topbar-view-active" : ""}`}
          onClick={toggleTo("spec")}
          aria-pressed={view === "spec"}
        >
          {view === "spec" ? "← Task chat" : "Specializations"}
        </button>
        <button
          type="button"
          className={`topbar-view-toggle${view === "shapes" ? " topbar-view-active" : ""}`}
          onClick={toggleTo("shapes")}
          aria-pressed={view === "shapes"}
          title="The plan shapes the planner selects — set each shape's iteration ceiling"
        >
          {view === "shapes" ? "← Task chat" : "Shapes"}
        </button>
        {hasChat && view === "tasks" && (
          <span
            className={`stream-pill stream-${streamStatus}`}
            role="status"
            aria-live="polite"
          >
            <span className="stream-dot" aria-hidden="true" />
            {STREAM_LABEL[streamStatus]}
          </span>
        )}
        <button
          type="button"
          className="theme-toggle"
          onClick={toggle}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
        >
          {theme === "dark" ? "☾" : "☀"}
        </button>
      </div>
    </header>
  );
}
