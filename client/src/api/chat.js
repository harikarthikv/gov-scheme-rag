// Government Scheme Finder — API client

const BASE = "http://localhost:8000/api";

export async function sendMessageStream(messages, sessionId, userId, onChunk, onMetadata, onThinking) {
  const response = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      session_id: sessionId,
      user_id: userId,     // C3: backend now verifies session ownership
    }),
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail?.detail ?? "Failed to connect to chat API");
  }

  const reader  = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer    = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n\n");
    buffer = lines.pop(); // keep incomplete chunk

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      try {
        const data = JSON.parse(line.substring(6));
        if      (data.type === "metadata" && onMetadata)  onMetadata(data.profile, data.schemes);
        else if (data.type === "chunk"    && onChunk)     onChunk(data.text);
        else if (data.type === "thinking" && onThinking)  onThinking(data.text);
        else if (data.type === "done")                    return;
      } catch (e) {
        console.error("SSE parse error:", e);
      }
    }
  }
}
