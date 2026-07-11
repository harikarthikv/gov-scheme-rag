import { useState, useEffect, useRef } from "react";
import Layout from "./components/Layout";
import AuthPage from "./components/AuthPage";
import MessageBubble from "./components/MessageBubble";
import TypingIndicator from "./components/TypingIndicator";
import { sendMessageStream } from "./api/chat";
import { registerUser, loginUser } from "./api/auth";
import { createSession, getSessions, getSessionMessages } from "./api/sessions";

const INITIAL_MESSAGE = {
  id: "initial",
  role: "assistant",
  content: "Hello! I can help you find Indian government schemes you are eligible for.\nTell me a bit about yourself — your age, where you are from, what you do, or anything else you think is relevant.",
};

function makeId() {
  return typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);
}

export default function App() {
  const [user,            setUser]            = useState(null);
  const [sessions,        setSessions]        = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);

  const [messages, setMessages] = useState([INITIAL_MESSAGE]);
  const [schemes,  setSchemes]  = useState(null);
  const [profile,  setProfile]  = useState(null);

  const [isTyping, setIsTyping] = useState(false);
  const [input,    setInput]    = useState("");
  const [error,    setError]    = useState(null);

  const bottomRef = useRef(null);
  const inputRef  = useRef(null);

  // Smooth-scroll to bottom whenever messages or typing state changes
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  // ── Auth ────────────────────────────────────────────────────────────────
  // Shown when user === null (H3: no hardcoded credentials)
  async function handleAuth(u) {
    setUser(u);
    const userSessions = await getSessions(u.user_id);
    setSessions(userSessions);
    if (userSessions.length > 0) {
      loadSession(userSessions[0].id, u.user_id);
    } else {
      handleNewChat(u.user_id);
    }
  }

  // M6: proper logout — clear all state
  function handleLogout() {
    setUser(null);
    setSessions([]);
    setMessages([INITIAL_MESSAGE]);
    setActiveSessionId(null);
    setSchemes(null);
    setProfile(null);
    setError(null);
    setInput("");
  }

  // ── Session management ──────────────────────────────────────────────────
  async function loadSession(sessionId, userId = user?.user_id) {
    if (!userId) return;
    setActiveSessionId(sessionId);
    try {
      const msgs = await getSessionMessages(sessionId, userId);
      // M5: attach stable ids to loaded messages
      const withIds = msgs.map(m => ({ ...m, id: makeId() }));
      setMessages(withIds.length > 0 ? withIds : [INITIAL_MESSAGE]);
      setSchemes(null);
      setProfile(null);
    } catch (e) {
      console.error(e);
    }
  }

  async function handleNewChat(userId = user?.user_id) {
    if (!userId) return;
    try {
      const res = await createSession(userId, "New Conversation");
      setActiveSessionId(res.session_id);
      setMessages([INITIAL_MESSAGE]);
      setSchemes(null);
      setProfile(null);
      const userSessions = await getSessions(userId);
      setSessions(userSessions);
    } catch (e) {
      console.error(e);
    }
  }

  // ── Send message ────────────────────────────────────────────────────────
  async function handleSend() {
    const text = input.trim();
    if (!text || isTyping || !activeSessionId) return;

    // M5: stable message id
    const userMsg = { id: makeId(), role: "user", content: text };
    const updatedMessages = [...messages, userMsg];

    setMessages(updatedMessages);
    setInput("");
    setIsTyping(true);
    setError(null);

    // Placeholder assistant message with stable id
    const assistantId = makeId();
    setMessages(prev => [...prev, { id: assistantId, role: "assistant", content: "", thinking: "" }]);

    try {
      await sendMessageStream(
        updatedMessages,
        activeSessionId,
        user?.user_id,
        // onChunk — M4: immutable update
        (chunkText) => {
          setMessages(prev =>
            prev.map(msg =>
              msg.id === assistantId
                ? { ...msg, content: msg.content + chunkText }
                : msg
            )
          );
          setIsTyping(false);
        },
        // onMetadata
        (metaProfile, metaSchemes) => {
          setProfile(metaProfile);
          setSchemes(metaSchemes);
        },
        // onThinking — M4: immutable update
        (thinkText) => {
          setMessages(prev =>
            prev.map(msg =>
              msg.id === assistantId
                ? { ...msg, thinking: (msg.thinking || "") + thinkText }
                : msg
            )
          );
          setIsTyping(false);
        }
      );
    } catch (err) {
      console.error(err);
      setError("Something went wrong. Please check the server and try again.");
      setMessages(prev =>
        prev.map(msg =>
          msg.id === assistantId
            ? { ...msg, content: "Sorry, I ran into an issue. Please try again in a moment." }
            : msg
        )
      );
    } finally {
      setIsTyping(false);
      inputRef.current?.focus();
    }
  }

  // ── Render ──────────────────────────────────────────────────────────────
  // H3: show auth screen when not logged in
  if (!user) {
    return <AuthPage onAuth={handleAuth} />;
  }

  return (
    <Layout
      user={user}
      sessions={sessions}
      activeSessionId={activeSessionId}
      onSelectSession={(id) => loadSession(id)}
      onNewChat={() => handleNewChat()}
      onLogout={handleLogout}
      schemes={schemes}
      profile={profile}
    >
      <div className="flex flex-col h-full bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 w-full overflow-hidden">
        {/* Message list */}
        <main className="flex-1 overflow-y-auto scrollbar-thin px-4 py-6 w-full">
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg) => (
              // M5: stable key — msg.id instead of array index
              <MessageBubble key={msg.id} role={msg.role} content={msg.content} thinking={msg.thinking} />
            ))}

            <TypingIndicator isTyping={isTyping} />

            {error && (
              <div className="bg-red-500/10 border border-red-500/25 text-red-400 text-sm rounded-xl px-4 py-3">
                {error}
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </main>

        {/* Input bar */}
        <footer className="flex-shrink-0 border-t border-slate-800/60 bg-slate-900/80 backdrop-blur-md px-4 py-4 w-full">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-end gap-3 bg-slate-800/80 border border-slate-700/60 rounded-2xl px-4 py-3 focus-within:border-indigo-500/60 focus-within:shadow-lg focus-within:shadow-indigo-500/10 transition-all duration-200">
              <textarea
                ref={inputRef}
                rows={1}
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  e.target.style.height = "auto";
                  e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder="Tell me about yourself — age, state, occupation, income…"
                disabled={isTyping}
                className="flex-1 bg-transparent text-slate-100 placeholder-slate-500 text-sm resize-none outline-none leading-relaxed disabled:opacity-50 min-h-[24px] max-h-[120px] overflow-y-auto scrollbar-thin"
                style={{ height: "24px" }}
              />
              <button
                id="send-message-btn"
                onClick={handleSend}
                disabled={!input.trim() || isTyping}
                className="flex-shrink-0 w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-md hover:shadow-indigo-500/30 transition-all duration-150 disabled:opacity-40"
              >
                <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </div>
            <p className="text-slate-600 text-xs text-center mt-2">
              Press Enter to send · Shift+Enter for new line
            </p>
          </div>
        </footer>
      </div>
    </Layout>
  );
}
