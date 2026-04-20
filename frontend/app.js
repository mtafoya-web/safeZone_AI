// SafeZone AI — app.js

// ── State ──────────────────────────────────────────────
let userLocation = null;
let wildfireData = [];
let nearestZone  = null;

// ── Init ───────────────────────────────────────────────
async function init() {
  await Promise.all([requestLocation(), loadCSV()]);

  if (userLocation?.lat !== null && wildfireData.length) {
    nearestZone = findNearestZone(userLocation.lat, userLocation.lng);
    showRiskBadge(nearestZone);
  }

  sendWelcomeMessage();
}

// ── Geocoding ──────────────────────────────────────────
async function geocode(query) {
  try {
    const res = await fetch(
      `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query)}&format=json&limit=1`,
      { headers: { 'Accept-Language': 'en' } }
    );
    const results = await res.json();
    if (!results.length) return null;
    const { lat, lon, display_name } = results[0];
    const label = display_name.split(',').slice(0, 2).join(',').trim();
    return { lat: parseFloat(lat), lng: parseFloat(lon), label };
  } catch { return null; }
}

async function reverseGeocode(lat, lng) {
  try {
    const res = await fetch(
      `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lng}&format=json`,
      { headers: { 'Accept-Language': 'en' } }
    );
    const data = await res.json();
    const a = data.address || {};
    return a.city || a.town || a.village || a.county || a.state || `${lat.toFixed(2)}°N`;
  } catch { return `${lat.toFixed(2)}°N`; }
}

// ── Location ───────────────────────────────────────────
function requestLocation() {
  return new Promise((resolve) => {
    if (!navigator.geolocation) { resolve(); return; }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLocation(
          pos.coords.latitude, pos.coords.longitude,
          `${pos.coords.latitude.toFixed(2)}°N, ${Math.abs(pos.coords.longitude).toFixed(2)}°W`
        );
        resolve();
      },
      (err) => { console.warn('Geolocation:', err.message); resolve(); },
      { timeout: 8000, maximumAge: 0 }
    );
  });
}

function setLocation(lat, lng, label) {
  userLocation = { lat, lng, label };
  document.getElementById('location-text').textContent = label;
  if (wildfireData.length && lat !== null) {
    nearestZone = findNearestZone(lat, lng);
    showRiskBadge(nearestZone);
  }
}

// ── Location Dropdown ──────────────────────────────────
const trigger  = document.getElementById('location-trigger');
const dropdown = document.getElementById('location-dropdown');

trigger.addEventListener('click', (e) => {
  e.stopPropagation();
  dropdown.classList.toggle('hidden');
  if (!dropdown.classList.contains('hidden')) {
    document.getElementById('location-input').focus();
  }
});

document.addEventListener('click', () => dropdown.classList.add('hidden'));
dropdown.addEventListener('click', (e) => e.stopPropagation());

document.getElementById('gps-btn').addEventListener('click', () => {
  const btn = document.getElementById('gps-btn');
  btn.textContent = '⏳ Locating...';
  btn.disabled = true;

  navigator.geolocation.getCurrentPosition(
    (pos) => {
      const label = `${pos.coords.latitude.toFixed(2)}°N, ${Math.abs(pos.coords.longitude).toFixed(2)}°W`;
      setLocation(pos.coords.latitude, pos.coords.longitude, label);
      dropdown.classList.add('hidden');
      btn.textContent = '🎯 Use My Current Location';
      btn.disabled = false;
      sendWelcomeMessage();
    },
    () => {
      btn.textContent = '❌ Access denied — try manual';
      setTimeout(() => { btn.textContent = '🎯 Use My Current Location'; btn.disabled = false; }, 2500);
    },
    { timeout: 8000, maximumAge: 0 }
  );
});

document.getElementById('location-submit-btn').addEventListener('click', async () => {
  const val = document.getElementById('location-input').value.trim();
  if (!val) return;

  const goBtn = document.getElementById('location-submit-btn');
  goBtn.textContent = '...';
  goBtn.disabled = true;

  const result = await geocode(val);
  if (result) {
    setLocation(result.lat, result.lng, result.label);
  } else {
    setLocation(null, null, val);
  }

  dropdown.classList.add('hidden');
  document.getElementById('location-input').value = '';
  goBtn.textContent = 'Go';
  goBtn.disabled = false;
  sendWelcomeMessage();
});

document.getElementById('location-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') document.getElementById('location-submit-btn').click();
});

// ── CSV ────────────────────────────────────────────────
async function loadCSV() {
  try {
    const res = await fetch('data/fire_data.csv');
    if (!res.ok) return;
    const text = await res.text();
    wildfireData = parseCSV(text);
    window.wildfireData = wildfireData;
  } catch (e) {
    console.warn('CSV not loaded:', e);
  }
}

function parseCSV(text) {
  const lines = text.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
  return lines.slice(1).map(line => {
    const vals = line.split(',');
    const row = {};
    headers.forEach((h, i) => row[h] = (vals[i] || '').trim());
    return row;
  });
}

function findNearestZone(lat, lng) {
  let nearest = null, minDist = Infinity;
  wildfireData.forEach(row => {
    const rLat = parseFloat(row.latitude);
    const rLng = parseFloat(row.longitude);
    if (isNaN(rLat) || isNaN(rLng)) return;
    const dist = Math.hypot(lat - rLat, lng - rLng);
    if (dist < minDist) { minDist = dist; nearest = row; }
  });
  return nearest;
}

// ── Risk Badge ─────────────────────────────────────────
function showRiskBadge(zone) {
  if (!zone) return;
  const risk = (zone.risk_level || '').toUpperCase();
  const badge = document.getElementById('risk-badge');
  badge.textContent = `${risk} RISK`;
  badge.className = 'risk-badge ' + risk.toLowerCase();
  badge.classList.remove('hidden');
}

// ── Chat UI ────────────────────────────────────────────
function appendBubble(text, role) {
  const messages = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = `bubble ${role}`;
  div.textContent = text;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

function showTyping() {
  const messages = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'typing-indicator';
  div.id = 'typing';
  div.innerHTML = '<span></span><span></span><span></span>';
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

function removeTyping() {
  document.getElementById('typing')?.remove();
}

// ── AI Context ─────────────────────────────────────────
async function buildEvacContext() {
  if (!wildfireData.length) return '';

  const highs = wildfireData.filter(r => (r.risk_level || '').toLowerCase() === 'high');
  const lows  = wildfireData
    .filter(r => (r.risk_level || '').toLowerCase() === 'low')
    .sort((a, b) => parseFloat(a.probability || 0) - parseFloat(b.probability || 0));

  const avgLat = highs.reduce((s, z) => s + parseFloat(z.latitude), 0) / (highs.length || 1);
  const avgLng = highs.reduce((s, z) => s + parseFloat(z.longitude), 0) / (highs.length || 1);

  const [highCity, ...safeCities] = await Promise.all([
    reverseGeocode(avgLat, avgLng),
    ...lows.slice(0, 3).map(z => reverseGeocode(parseFloat(z.latitude), parseFloat(z.longitude))),
  ]);

  return `\n\nEvacuation data:\n- Danger concentrated near: ${highCity}\n- Safest areas: ${safeCities.join(', ')}\n- Recommend evacuating away from ${highCity} toward the safe areas.`;
}

async function buildSystemContext() {
  const loc = userLocation?.label || 'Unknown location';
  const highCount = wildfireData.filter(r => (r.risk_level || '').toLowerCase() === 'high').length;
  const medCount  = wildfireData.filter(r => (r.risk_level || '').toLowerCase() === 'medium').length;

  let zoneInfo = 'No zone data available.';
  if (nearestZone) {
    const z = nearestZone;
    zoneInfo = [
      `Risk: ${z.risk_level || 'Unknown'}`,
      `Advisory: ${z.advisory_level || 'N/A'}`,
      `Probability: ${z.probability ? (parseFloat(z.probability) * 100).toFixed(1) + '%' : 'N/A'}`,
      `Fire count: ${z.fire_count || 0}`,
      `Wind: ${z.wind_speed || 'N/A'} mph`,
      `Forecast: ${z.is_forecast === '1' ? 'Yes' : 'No'}`,
    ].join(', ');
  } else if (wildfireData.length) {
    zoneInfo = `No exact zone match. Dataset has ${highCount} HIGH and ${medCount} MEDIUM risk zones — use your knowledge of the user's city to give relevant advice.`;
  }

  return `You are a wildfire safety expert and emergency response assistant for a coastal community. Wildfires here threaten lives, homes, and the ocean ecosystem through ash runoff and air quality degradation.

Situational data:
- User location: ${loc}
- Nearest zone: ${zoneInfo}
- Active HIGH risk zones: ${highCount}
- Active MEDIUM risk zones: ${medCount}

Rules:
- Prioritize safety — HIGH risk or probability >50% means recommend evacuation
- Give specific, actionable advice using place names people recognize
- Mention coastal/ocean ecosystem impact where relevant
- Never use raw lat/lng — use city names, neighborhoods, or highways
- Cite CAL FIRE, local emergency services, or Ready.gov when helpful
- Keep responses concise${await buildEvacContext()}`;
}

// ── AI Call ────────────────────────────────────────────
async function callAI(userMessage) {
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${window.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: 'gpt-4.1-nano',
      messages: [
        { role: 'system', content: await buildSystemContext() },
        { role: 'user',   content: userMessage },
      ],
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.error?.message || `HTTP ${res.status}`);
  }
  const data = await res.json();
  return data.choices?.[0]?.message?.content || 'No response.';
}

// ── SMS Detection ──────────────────────────────────────
function extractPhone(text) {
  const match = text.match(/(\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})/);
  if (!match) return null;
  const digits = match[1].replace(/\D/g, '');
  if (digits.length === 11 && digits.startsWith('1')) return '+' + digits;
  if (digits.length === 10) return '+1' + digits;
  return null;
}

async function sendCallFromChat(phone, userMessage) {
  const callPrompt = `The user asked: "${userMessage}". Write a concise wildfire safety alert (max 3 sentences) for the location they mentioned or their current area (${userLocation?.label || 'unknown'}). Include risk level, key advisory, and one action to take. Plain text only, no markdown.`;

  const callBody = await callAI(callPrompt);

  const zone = nearestZone || {};
  const res = await fetch('http://localhost:8001/api/alert/call', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      phone,
      risk_level: zone.risk_level || 'Unknown',
      advisory: callBody,
      location: userLocation?.label || 'your area',
    }),
  });

  if (!res.ok) throw new Error('Call failed');
  return phone;
}

async function sendMessage(text) {
  if (!text.trim()) return;

  const input = document.getElementById('chat-input');
  const btn   = document.getElementById('send-btn');

  input.value = '';
  input.disabled = true;
  btn.disabled = true;

  appendBubble(text, 'user');
  showTyping();

  try {
    const phone = extractPhone(text);
    if (phone) {
      // Voice call flow
      await sendCallFromChat(phone, text);
      removeTyping();
      appendBubble(`✅ Calling ${phone} now! You'll receive a voice alert with current wildfire risk and safety advice for your area.`, 'ai');
    } else {
      // Normal chat flow
      appendBubble(await callAI(text), 'ai');
    }
  } catch (e) {
    removeTyping();
    appendBubble('Unable to reach AI or send SMS. Check your connection.', 'error');
  } finally {
    removeTyping();
    input.disabled = false;
    btn.disabled = false;
    input.focus();
  }
}

async function sendWelcomeMessage() {
  if (!userLocation) return;
  const risk     = nearestZone?.risk_level || 'unknown';
  const advisory = nearestZone?.advisory_level || '';
  const prob     = nearestZone ? (parseFloat(nearestZone.probability || 0) * 100).toFixed(1) : null;
  const forecast = nearestZone?.is_forecast === '1' ? 'Yes' : 'No';

  const prompt = `Welcome the user in ${userLocation.label}. Wildfire risk: ${risk}${advisory ? ', advisory: ' + advisory : ''}${prob ? ', fire probability: ' + prob + '%' : ''}. Next-day forecast: ${forecast}. Summarize what they should know and do. Mention coastal/ocean ecosystem impact. Keep it to 3–4 sentences.`;

  showTyping();
  try {
    appendBubble(await callAI(prompt), 'ai');
  } catch {
    appendBubble('Welcome to SafeZone AI. Unable to load risk summary — check your API key.', 'error');
  } finally {
    removeTyping();
  }
}

// ── Analyze Map ────────────────────────────────────────
document.getElementById('analyze-btn').addEventListener('click', async () => {
  const btn = document.getElementById('analyze-btn');
  btn.textContent = '⏳ Analyzing...';
  btn.disabled = true;

  const highs = wildfireData.filter(r => (r.risk_level || '').toLowerCase() === 'high');
  const meds  = wildfireData.filter(r => (r.risk_level || '').toLowerCase() === 'medium');
  const lows  = wildfireData.filter(r => (r.risk_level || '').toLowerCase() === 'low');

  const topZoneNames = await Promise.all(
    [...highs].sort((a, b) => parseFloat(b.probability || 0) - parseFloat(a.probability || 0))
      .slice(0, 5)
      .map(async z => {
        const city = await reverseGeocode(parseFloat(z.latitude), parseFloat(z.longitude));
        return `${city} — ${(parseFloat(z.probability||0)*100).toFixed(1)}% probability, wind ${z.wind_speed} mph`;
      })
  );

  const safeNames = await Promise.all(
    lows.slice(0, 3).map(z => reverseGeocode(parseFloat(z.latitude), parseFloat(z.longitude)))
  );

  const prompt = `Analyze this wildfire situation:
- HIGH risk zones: ${highs.length}, MEDIUM: ${meds.length}, LOW: ${lows.length}
- Top danger areas: ${topZoneNames.join('; ')}
- Safest areas: ${safeNames.join(', ')}
- User location: ${userLocation?.label || 'unknown'}

Give a 4–5 sentence situational briefing using place names. Where is danger concentrated, what areas are safest, and what should people do now?`;

  await sendMessage(prompt);
  btn.textContent = '🔍 Analyze Map';
  btn.disabled = false;
});

// ── Event Listeners ────────────────────────────────────
document.getElementById('send-btn').addEventListener('click', () => {
  sendMessage(document.getElementById('chat-input').value);
});

document.getElementById('chat-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendMessage(e.target.value);
});

document.querySelectorAll('.suggestion-chip').forEach(chip => {
  chip.addEventListener('click', () => sendMessage(chip.textContent.trim()));
});

// ── Start ──────────────────────────────────────────────
init();
