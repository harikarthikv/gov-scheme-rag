import SchemeCard from "./SchemeCard";

export default function RightSidebar({ isOpen, setIsOpen, schemes, profile }) {
  return (
    <aside className={`
      fixed inset-y-0 right-0 z-50 w-80 bg-slate-900 border-l border-slate-800/60
      transform transition-transform duration-300 ease-in-out lg:relative lg:translate-x-0
      flex flex-col
      ${isOpen ? 'translate-x-0' : 'translate-x-full'}
    `}>
      <div className="p-4 border-b border-slate-800/60 flex items-center justify-between">
        <h2 className="text-slate-200 font-bold tracking-tight flex items-center gap-2">
          <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
          Recommendations
        </h2>
        <button className="lg:hidden text-slate-400" onClick={() => setIsOpen(false)}>
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-4">
        {profile && Object.keys(profile).length > 0 && (
          <div className="mb-6">
            <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Extracted Profile</div>
            <div className="flex flex-wrap gap-2">
              {Object.entries(profile).map(([k, v]) => (
                <div key={k} className="bg-slate-800/50 border border-slate-700 rounded-md px-2 py-1 text-xs text-slate-300">
                  <span className="text-slate-500 mr-1">{k.replaceAll("_", " ")}:</span>
                  {typeof v === 'boolean' ? (v ? 'Yes' : 'No') : v}
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Matched Schemes</div>
        
        {!schemes ? (
          <div className="text-center py-8">
            <p className="text-slate-500 text-sm">Tell me about yourself to see schemes you qualify for.</p>
          </div>
        ) : schemes.length === 0 ? (
          <div className="text-center py-8 border border-dashed border-slate-700/50 rounded-xl bg-slate-800/20">
            <p className="text-slate-400 text-sm">No schemes found yet.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {schemes.map((scheme, i) => (
              <SchemeCard
                key={i}
                name={scheme.name}
                eligibility_status={scheme.eligibility_status}
                reason={scheme.reason}
                key_benefit={scheme.key_benefit}
                url={scheme.url}
              />
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}
