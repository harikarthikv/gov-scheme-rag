# Streaming Architecture (Server-Sent Events)

To provide a modern, low-latency conversational experience, the application utilizes Server-Sent Events (SSE) to stream responses from the backend to the frontend.

## Why SSE?
Unlike WebSockets (which are bidirectional and complex to scale) or standard HTTP POSTs (which force the user to stare at a loading spinner for 10+ seconds while the LLM generates a response), SSE is a unidirectional streaming protocol built on standard HTTP. 

It allows the FastAPI server to keep the connection open and `yield` chunks of data as soon as they are available.

## The Data Stream Lifecycle

When `POST /api/chat` is hit, the FastAPI endpoint returns a `StreamingResponse`.

### 1. Metadata Yield
As soon as the Profile Extraction and Eligibility Matching chains complete (usually within 2-4 seconds), the backend immediately yields a `metadata` event:
```json
data: {"type": "metadata", "profile": {"age": 30}, "schemes": [{"name": "Scheme A"}]}
```
The React frontend instantly intercepts this, populates the Right Sidebar, and renders the scheme cards *before* the conversational reply has even begun generating.

### 2. Token Yielding
The backend then prompts the LLM to generate the conversational reply. As tokens arrive from the Gemini API (or Ollama), they are immediately yielded to the client:
```json
data: {"type": "chunk", "text": "Hello"}
data: {"type": "chunk", "text": " there!"}
```
The React frontend appends these chunks to the `MessageBubble` state in real-time, creating the typing effect.

### 3. Reasoning Tokens (Local Fallback)
If the system falls back to the local `qwen3:0.6b` model, the model emits reasoning tokens inside `<think>...</think>` tags before generating the final answer. 
The SSE parser intercepts these tags and yields them as a special event:
```json
data: {"type": "thinking", "text": "The user is 30, so..."}
```
The frontend renders these inside a collapsible "Reasoning" block above the final message.

### 4. Completion Yield
Once the LLM finishes, a final event is sent:
```json
data: {"type": "done"}
```
The frontend closes the stream and removes the typing indicator. The backend then safely persists the fully generated response to the SQLite database.
