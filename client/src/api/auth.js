import axios from "axios";

const BASE = "http://localhost:8000/api";

export async function registerUser(email, password) {
  const res = await axios.post(`${BASE}/register`, { email, password });
  return res.data;
}

export async function loginUser(email, password) {
  const res = await axios.post(`${BASE}/login`, { email, password });
  return res.data;
}
