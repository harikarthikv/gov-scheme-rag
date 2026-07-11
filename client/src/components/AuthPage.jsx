// Simple login / register page — replaces the hardcoded dummy user (H3)
import { useState } from "react";
import { registerUser, loginUser } from "../api/auth";

export default function AuthPage({ onAuth }) {
  const [mode, setMode]         = useState("login"); // "login" | "register"
  const [email, setEmail]       = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const u = mode === "register"
        ? await registerUser(email, password)
        : await loginUser(email, password);
      onAuth(u);
    } catch {
      setError(
        mode === "register"
          ? "Registration failed — that email may already be in use."
          : "Invalid email or password."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Branding */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 shadow-xl mb-4">
            <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M8 14v3m4-3v3m4-3v3M3 21h18M3 10h18M3 7l9-4 9 4M4 10h16v11H4V10z" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-slate-100">Scheme Finder</h1>
          <p className="text-slate-400 text-sm mt-1">
            Discover Indian government schemes you qualify for
          </p>
        </div>

        {/* Card */}
        <div className="bg-slate-800/60 border border-slate-700/50 backdrop-blur-sm rounded-2xl p-8 shadow-2xl">
          <h2 className="text-lg font-semibold text-slate-100 mb-6">
            {mode === "login" ? "Sign in" : "Create an account"}
          </h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5" htmlFor="auth-email">
                Email address
              </label>
              <input
                id="auth-email"
                type="email"
                required
                autoComplete="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                className="w-full bg-slate-700/60 border border-slate-600/50 rounded-xl px-4 py-2.5 text-slate-100 placeholder-slate-500 text-sm outline-none focus:border-indigo-500/70 focus:ring-2 focus:ring-indigo-500/20 transition-all"
                placeholder="you@example.com"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5" htmlFor="auth-password">
                Password
              </label>
              <input
                id="auth-password"
                type="password"
                required
                autoComplete={mode === "register" ? "new-password" : "current-password"}
                value={password}
                onChange={e => setPassword(e.target.value)}
                className="w-full bg-slate-700/60 border border-slate-600/50 rounded-xl px-4 py-2.5 text-slate-100 placeholder-slate-500 text-sm outline-none focus:border-indigo-500/70 focus:ring-2 focus:ring-indigo-500/20 transition-all"
                placeholder={mode === "register" ? "Min. 6 characters" : "Your password"}
              />
            </div>

            {error && (
              <p className="text-red-400 text-xs bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 text-white font-semibold text-sm shadow-lg hover:shadow-indigo-500/30 transition-all duration-150 disabled:opacity-50"
            >
              {loading ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
            </button>
          </form>

          <p className="text-center text-xs text-slate-500 mt-5">
            {mode === "login" ? "Don't have an account? " : "Already have an account? "}
            <button
              onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(""); }}
              className="text-indigo-400 hover:text-indigo-300 font-medium transition-colors"
            >
              {mode === "login" ? "Register" : "Sign in"}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}
