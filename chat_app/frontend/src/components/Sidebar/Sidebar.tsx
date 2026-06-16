import { useChats } from "../../api/queries";
import type { ChatRecord } from "../../api/types";
import "./Sidebar.css";

interface SidebarProps {
  selectedId: string | null;
  onSelect: (chatId: string) => void;
  onNewChat: () => void;
  creating: boolean;
}

export function Sidebar({ selectedId, onSelect, onNewChat, creating }: SidebarProps) {
  const chats = useChats();

  return (
    <nav className="sidebar" aria-label="Chats">
      <button type="button" className="sidebar-new" onClick={onNewChat} disabled={creating}>
        <span aria-hidden="true">+</span> {creating ? "Creating…" : "New chat"}
      </button>

      <div className="sidebar-list" role="list">
        {chats.isLoading && <p className="sidebar-hint">Loading chats…</p>}
        {chats.isError && <p className="sidebar-hint sidebar-error">Could not load chats.</p>}
        {chats.data?.length === 0 && !chats.isLoading && (
          <p className="sidebar-hint">No chats yet. Start one above.</p>
        )}
        {chats.data?.map((chat) => (
          <ChatRow
            key={chat.chat_id}
            chat={chat}
            active={chat.chat_id === selectedId}
            onSelect={() => onSelect(chat.chat_id)}
          />
        ))}
      </div>
    </nav>
  );
}

function ChatRow({
  chat,
  active,
  onSelect,
}: {
  chat: ChatRecord;
  active: boolean;
  onSelect: () => void;
}) {
  const turnCount = chat.turns.length;
  return (
    <button
      type="button"
      role="listitem"
      className={`chat-row${active ? " chat-row-active" : ""}`}
      onClick={onSelect}
      aria-current={active ? "true" : undefined}
    >
      <span className="chat-row-title">{chat.title || "Untitled chat"}</span>
      <span className="chat-row-meta">
        {turnCount} {turnCount === 1 ? "turn" : "turns"}
      </span>
    </button>
  );
}
