import { useCallback, useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Sidebar } from "./components/Sidebar/Sidebar";
import { ConversationPane } from "./components/Conversation/ConversationPane";
import { DagView } from "./components/Dag/DagView";
import { ArtifactsPanel } from "./components/Artifacts/ArtifactsPanel";
import { TopBar, type AppView } from "./components/TopBar/TopBar";
import { CotOverlay } from "./components/CotOverlay/CotOverlay";
import { SpecChatSurface } from "./components/SpecChat/SpecChatSurface";
import { ShapesSurface } from "./components/Shapes/ShapesSurface";
import { TunedExplainer } from "./components/TunedExplainer/TunedExplainer";
import { useChat, useCreateChat, queryKeys } from "./api/queries";
import { useTaskRun } from "./hooks/useTaskRun";
import "./App.css";

export function App() {
  const [chatId, setChatId] = useState<string | null>(null);
  const [view, setView] = useState<AppView>("tasks");
  const qc = useQueryClient();

  const chatQuery = useChat(chatId);
  const createChat = useCreateChat();
  const run = useTaskRun(chatId);

  // When a run finishes, refresh the persisted history + artifacts for this chat.
  const completed = run.lastCompletedRunId;
  useEffect(() => {
    if (completed && chatId) {
      void qc.invalidateQueries({ queryKey: queryKeys.chat(chatId) });
      void qc.invalidateQueries({ queryKey: queryKeys.chats });
    }
  }, [completed, chatId, qc]);

  const handleNewChat = useCallback(() => {
    createChat.mutate(undefined, { onSuccess: (rec) => setChatId(rec.chat_id) });
  }, [createChat]);

  const backToTasks = useCallback(() => setView("tasks"), []);

  return (
    <div className="app-shell">
      <TopBar
        streamStatus={run.streamStatus}
        hasChat={chatId !== null}
        view={view}
        onSetView={setView}
      />
      {view === "spec" ? (
        <div className="app-body app-body-single">
          <SpecChatSurface onClose={backToTasks} />
        </div>
      ) : view === "shapes" ? (
        <div className="app-body app-body-single">
          <ShapesSurface onClose={backToTasks} />
        </div>
      ) : view === "tuned" ? (
        <div className="app-body app-body-single">
          <TunedExplainer onClose={backToTasks} />
        </div>
      ) : (
        <div className="app-body">
          <Sidebar
            selectedId={chatId}
            onSelect={setChatId}
            onNewChat={handleNewChat}
            creating={createChat.isPending}
          />
          <main className="app-main" aria-label="Task chat">
            <ConversationPane
              chat={chatQuery.data ?? null}
              loading={chatQuery.isLoading}
              busy={run.busy}
              runError={run.runError}
              canSend={chatId !== null && !run.busy}
              onSend={run.send}
              onNewChat={handleNewChat}
              pendingMessage={run.pendingMessage}
              pendingResolution={run.pendingResolution}
              pendingClarification={run.pendingClarification}
              onResume={run.resume}
              onResolveClarification={run.resolveClarification}
              resuming={run.resuming}
              resumeError={run.resumeError}
            />
          </main>
          <aside className="app-aside" aria-label="Plan and artifacts">
            <DagView nodes={run.nodes} runStatus={run.runStatus} />
            <ArtifactsPanel
              runArtifacts={run.artifacts}
              chatArtifacts={chatQuery.data?.artifacts ?? []}
            />
          </aside>
        </div>
      )}

      {/* (a) chain-of-thought brain-icon overlay — a fixed corner pop-up, only
          while a task chat is selected (it streams that chat's tool activity). */}
      {view === "tasks" && <CotOverlay chatId={chatId} />}
    </div>
  );
}
