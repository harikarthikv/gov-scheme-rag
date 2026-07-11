import axios from "axios";

const BASE = "http://localhost:8000/api";

export async function createSession(userId, title) {
  const res = await axios.post(`${BASE}/sessions`, { user_id: userId, title });
  return res.data;
}

export async function getSessions(userId) {
  const res = await axios.get(`${BASE}/sessions`, { params: { user_id: userId } });
  return res.data.sessions;
}

export async function getSessionMessages(sessionId, userId) {
  const res = await axios.get(`${BASE}/sessions/${sessionId}/messages`, { params: { user_id: userId } });
  return res.data.messages;
}
