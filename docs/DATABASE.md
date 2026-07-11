# Database Schema & Design

The application utilizes a local **SQLite** database (`server/rag_assistant.db`) for lightweight, zero-config data persistence. 

## Schema Design

### `users` Table
Stores authentication credentials.
- `id` (INTEGER PRIMARY KEY)
- `email` (TEXT UNIQUE)
- `password_hash` (TEXT): Salted Scrypt hash (Base64 encoded).

### `sessions` Table
Groups messages into distinct conversations.
- `id` (TEXT PRIMARY KEY): UUID.
- `user_id` (INTEGER): Foreign key to `users(id)`.
- `title` (TEXT): Default "New Conversation".
- `created_at` (TIMESTAMP)

### `messages` Table
The chronological ledger of the conversation.
- `id` (INTEGER PRIMARY KEY)
- `session_id` (TEXT): Foreign key to `sessions(id)`.
- `role` (TEXT): `user` or `assistant`.
- `content` (TEXT): The message text.
- `created_at` (TIMESTAMP)

## Design Decisions

1. **Why SQLite?**
   This is an MVP application focused on AI capabilities. SQLite provides a zero-setup, highly performant embedded database that requires no external Docker containers or cloud provisioning, making the project instantly reproducible.
2. **WAL Mode**:
   Write-Ahead Logging (`PRAGMA journal_mode=WAL`) is enabled on startup to allow concurrent reads and writes, preventing database locks during long-running LLM streams.
3. **No Auth Tokens**:
   To minimize complexity, the client stores the `user_id` in React state upon login. In a production environment, this should be replaced with HttpOnly JWT cookies.
