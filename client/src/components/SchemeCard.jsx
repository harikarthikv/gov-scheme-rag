// Renders a single matched scheme as a card
// Props: name, eligibility_status ("eligible" | "partial"), reason, key_benefit, url
// M9: URLs are validated before being used as href

function safeUrl(url) {
  try {
    const u = new URL(url);
    return u.protocol === "https:" || u.protocol === "http:" ? url : "#";
  } catch {
    return "#";
  }
}

export default function SchemeCard({ name, eligibility_status, reason, key_benefit, url }) {
  const isEligible = eligibility_status === "eligible";
  const href = safeUrl(url);

  return (
    <div className="bg-slate-800/60 border border-slate-700/50 backdrop-blur-sm rounded-2xl p-5 shadow-lg hover:shadow-indigo-500/10 hover:border-slate-600/70 transition-all duration-300 animate-fade-in-up group">
      {/* Header */}
      <div className="flex items-start justify-between gap-3 mb-4">
        <h3 className="text-slate-100 font-semibold text-sm leading-snug flex-1">{name}</h3>
        <span
          className={`flex-shrink-0 inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full ${
            isEligible
              ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
              : "bg-amber-500/15 text-amber-400 border border-amber-500/30"
          }`}
        >
          <span className={`w-1.5 h-1.5 rounded-full ${isEligible ? "bg-emerald-400" : "bg-amber-400"}`} />
          {isEligible ? "Eligible" : "Partial Match"}
        </span>
      </div>

      {/* Why you qualify */}
      <div className="mb-3">
        <p className="text-xs font-semibold text-indigo-400 uppercase tracking-wide mb-1">
          Why you qualify
        </p>
        <p className="text-slate-300 text-sm leading-relaxed">{reason}</p>
      </div>

      {/* Key benefit */}
      <div className="mb-4 bg-slate-700/40 rounded-xl p-3 border border-slate-600/30">
        <p className="text-xs font-semibold text-purple-400 uppercase tracking-wide mb-1">
          Key Benefit
        </p>
        <p className="text-slate-200 text-sm leading-relaxed">{key_benefit}</p>
      </div>

      {/* Apply link — M9: href validated */}
      {href !== "#" ? (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          id={`apply-link-${name?.replace(/\s+/g, "-").toLowerCase().slice(0, 30)}`}
          className="inline-flex items-center gap-2 text-indigo-400 hover:text-indigo-300 text-sm font-medium transition-colors duration-200 group-hover:gap-3"
        >
          Apply on MyScheme.gov.in
          <svg className="w-4 h-4 transition-transform duration-200 group-hover:translate-x-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
          </svg>
        </a>
      ) : (
        <span className="text-slate-500 text-xs italic">No direct link available</span>
      )}
    </div>
  );
}
