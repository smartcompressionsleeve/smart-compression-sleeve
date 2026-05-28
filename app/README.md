# Companion App

Flutter mobile companion app for the Smart Compression Sleeve. Displays
real-time exercise form feedback, counts reps, and persists session history
to a Supabase backend.

## Stack

- **Framework:** Flutter / Dart
- **Backend:** Supabase (Postgres + Auth, RLS-enabled)
- **Connection:** WebSocket to the local Python bridge for live sensor feed

## Setup

```bash
flutter pub get
flutter run
```

Make sure the Python bridge (`../bridge/`) is running and that your device
is on the same network if running the app on a physical phone.

## Configuration

Supabase credentials are loaded at runtime. **Do not commit `.env` files or
hardcoded service-role keys** — only the anon key is safe to ship, and even
that should be loaded from configuration, not embedded in source.
