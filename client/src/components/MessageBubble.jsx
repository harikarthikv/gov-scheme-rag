import { useState } from "react";
import ReactMarkdown from "react-markdown";

// Renders a single chat message bubble
// Props: role ("user" | "assistant"), content (string), thinking (string | undefined)
// Thinking content is shown in a collapsible block above the main reply

function ThinkingBlock({ text }) {
  const [open, setOpen] = useState(false);
  if (!text || !text.trim()) return null;

  return (
    <div className="mb-3">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-300 transition-colors"
      >
        {/* brain icon */}
        <svg className="w-3.5 h-3.5 text-indigo-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
        <span className="text-indigo-400 font-medium">Reasoning</span>
        <svg
          className={`w-3 h-3 transition-transform ${open ? "rotate-180" : ""}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="mt-2 pl-3 border-l-2 border-indigo-500/30 text-xs text-slate-400 leading-relaxed whitespace-pre-wrap font-mono max-h-60 overflow-y-auto scrollbar-thin">
          {text.trim()}
        </div>
      )}
    </div>
  );
}

export default function MessageBubble({ role, content, thinking }) {
  const isUser = role === "user";

  return (
    <div
      className={`flex items-start gap-3 ${isUser ? "flex-row-reverse animate-slide-in-right" : "animate-slide-in-left"}`}
    >
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center shadow-lg ${
          isUser
            ? "bg-gradient-to-br from-indigo-400 to-purple-500"
            : "bg-gradient-to-br from-indigo-500 to-purple-600"
        }`}
      >
        {isUser ? (
          <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
          </svg>
        ) : (
          <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
        )}
      </div>

      {/* Bubble */}
      <div
        className={`max-w-[75%] px-5 py-3.5 rounded-2xl shadow-md text-sm leading-relaxed ${
          isUser
            ? "bg-gradient-to-br from-indigo-600 to-purple-700 text-white rounded-tr-sm whitespace-pre-wrap"
            : "bg-slate-800/80 border border-slate-700/50 backdrop-blur-sm text-slate-100 rounded-tl-sm"
        }`}
      >
        {isUser ? (
          content
        ) : (
          <>
            {/* Collapsible thinking block — shown when model includes reasoning */}
            {thinking && (
              <ThinkingBlock text={thinking} />
            )}

            <ReactMarkdown
              className="prose prose-invert prose-sm max-w-none"
              components={{
                a: ({ node, ...props }) => (
                  <a {...props} target="_blank" rel="noopener noreferrer" className="text-indigo-400 hover:text-indigo-300 underline" />
                ),
                p: ({ node, ...props }) => <p {...props} className="mb-2 last:mb-0" />,
                ul: ({ node, ...props }) => <ul {...props} className="list-disc pl-4 mb-2 last:mb-0" />,
                ol: ({ node, ...props }) => <ol {...props} className="list-decimal pl-4 mb-2 last:mb-0" />,
                li: ({ node, ...props }) => <li {...props} className="mb-1" />,
              }}
            >
              {content}
            </ReactMarkdown>
          </>
        )}
      </div>
    </div>
  );
}
