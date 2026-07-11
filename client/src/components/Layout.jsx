import { useState, useEffect } from "react";
import LeftSidebar from "./LeftSidebar";
import RightSidebar from "./RightSidebar";

export default function Layout({ 
  children, 
  user,
  sessions, 
  activeSessionId, 
  onSelectSession, 
  onNewChat,
  onLogout,
  schemes,
  profile
}) {
  const [leftOpen, setLeftOpen] = useState(false);
  const [rightOpen, setRightOpen] = useState(false);

  // Close sidebars on mobile when selecting something
  useEffect(() => {
    setLeftOpen(false);
  }, [activeSessionId]);

  return (
    <div className="flex h-screen bg-slate-950 overflow-hidden">
      {/* Mobile overlays */}
      {(leftOpen || rightOpen) && (
        <div 
          className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-40 lg:hidden"
          onClick={() => { setLeftOpen(false); setRightOpen(false); }}
        />
      )}

      {/* Left Sidebar */}
      <LeftSidebar 
        isOpen={leftOpen} 
        setIsOpen={setLeftOpen} 
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={onSelectSession}
        onNewChat={onNewChat}
        user={user}
        onLogout={onLogout}
      />

      {/* Center Chat Area (the children) */}
      <main className="flex-1 flex flex-col h-full w-full relative min-w-0 z-10 transition-all">
        {/* Mobile Header for Sidebar toggles */}
        <header className="lg:hidden flex items-center justify-between p-4 border-b border-slate-800/60 bg-slate-900/80 backdrop-blur-md">
          <button 
            onClick={() => setLeftOpen(true)}
            className="p-2 -ml-2 text-slate-400 hover:text-slate-100"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          
          <div className="text-slate-200 font-bold text-lg">Scheme Finder</div>
          
          <button 
            onClick={() => setRightOpen(true)}
            className="p-2 -mr-2 text-slate-400 hover:text-slate-100 relative"
          >
            {schemes?.length > 0 && (
              <span className="absolute top-2 right-2 w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            )}
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
          </button>
        </header>
        
        {children}
      </main>

      {/* Right Sidebar */}
      <RightSidebar 
        isOpen={rightOpen} 
        setIsOpen={setRightOpen} 
        schemes={schemes}
        profile={profile}
      />
    </div>
  );
}
