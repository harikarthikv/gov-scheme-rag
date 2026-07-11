export default function LeftSidebar({ 
  isOpen, 
  setIsOpen, 
  sessions, 
  activeSessionId, 
  onSelectSession, 
  onNewChat,
  user,
  onLogout 
}) {
  return (
    <aside className={`
      fixed inset-y-0 left-0 z-50 w-72 bg-slate-900 border-r border-slate-800/60
      transform transition-transform duration-300 ease-in-out lg:relative lg:translate-x-0
      flex flex-col
      ${isOpen ? 'translate-x-0' : '-translate-x-full'}
    `}>
      <div className="p-4 border-b border-slate-800/60 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 14v3m4-3v3m4-3v3M3 21h18M3 10h18M3 7l9-4 9 4M4 10h16v11H4V10z" />
            </svg>
          </div>
          <span className="text-slate-100 font-bold tracking-tight">GovSchemes</span>
        </div>
        <button className="lg:hidden text-slate-400" onClick={() => setIsOpen(false)}>
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div className="p-4">
        <button 
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-100 py-2.5 rounded-xl border border-slate-700/60 transition-colors text-sm font-medium"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin px-3 py-2 space-y-1">
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 px-2">History</div>
        {sessions?.length > 0 ? sessions.map(session => (
          <button
            key={session.id}
            onClick={() => onSelectSession(session.id)}
            className={`
              w-full text-left px-3 py-2.5 rounded-lg text-sm transition-colors
              ${activeSessionId === session.id 
                ? 'bg-indigo-500/10 text-indigo-400 font-medium' 
                : 'text-slate-300 hover:bg-slate-800/50 hover:text-slate-200'}
            `}
          >
            <div className="truncate">{session.title}</div>
            <div className="text-xs text-slate-500 mt-0.5 opacity-60">
              {new Date(session.created_at).toLocaleDateString()}
            </div>
          </button>
        )) : (
          <div className="text-xs text-slate-500 px-2 italic">No chats yet.</div>
        )}
      </div>

      {user && (
        <div className="p-4 border-t border-slate-800/60">
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-400 truncate pr-2">{user.email}</span>
            <button onClick={onLogout} className="text-slate-500 hover:text-red-400 transition-colors">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
            </button>
          </div>
        </div>
      )}
    </aside>
  );
}
