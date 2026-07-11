# Frontend Architecture

The frontend is a Single Page Application (SPA) built to deliver a highly responsive, app-like conversational experience.

## Tech Stack
- **Framework**: React 18
- **Build Tool**: Vite (blazing fast HMR and optimized production builds)
- **Styling**: Tailwind CSS
- **Markdown Parsing**: `react-markdown`

## Component Structure

```text
client/src/
├── App.jsx                 (Main entry point & State Controller)
├── api/
│   ├── auth.js             (Login/Register Axios calls)
│   ├── chat.js             (SSE Streaming Fetch client)
│   └── sessions.js         (Session management API calls)
└── components/
    ├── AuthPage.jsx        (Login/Registration UI)
    ├── Layout.jsx          (Main application wrapper handling sidebars)
    ├── LeftSidebar.jsx     (Chat history & New Chat button)
    ├── RightSidebar.jsx    (Profile Extracted & Matched Schemes)
    ├── MessageBubble.jsx   (Renders individual user/assistant messages)
    ├── SchemeCard.jsx      (Renders an individual eligible scheme)
    └── TypingIndicator.jsx (Animated loading state)
```

## State Management

Instead of introducing heavy state management libraries like Redux or Zustand, the state is managed centrally at the top of the component tree in `App.jsx`.

- **`user`**: Tracks the authenticated user (enables the `AuthPage` guard).
- **`sessions`**: Array of the user's conversation history.
- **`messages`**: The active chat log. Sent to the backend on every new prompt.
- **`profile` & `schemes`**: Pushed from the backend via SSE metadata chunks. Propagated down to `RightSidebar.jsx`.

## Streaming Updates (Immutability)

Handling Server-Sent Events requires careful React state management to prevent race conditions or dropped chunks.
When a chunk arrives, the state is updated immutably using the functional form of `setMessages`:

```javascript
setMessages(prev =>
  prev.map(msg =>
    msg.id === assistantId
      ? { ...msg, content: msg.content + chunkText }
      : msg
  )
);
```

## Responsive Design
The application features a responsive three-pane layout. 
On desktop, both sidebars (History and Recommendations) are visible. 
On mobile devices, the sidebars collapse into hamburger menus using a slide-in overlay, ensuring the chat interface remains usable on smaller screens.
