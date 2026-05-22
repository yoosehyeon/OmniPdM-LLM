// CSRF 토큰 처리 — module-level singleton.
//
// 백엔드 /api/csrf 는 per-session token 을 재사용 (services/auth/sessions.py
// get_or_create_csrf_token). 만료 없음 — 세션 동안 같은 값 반환.
// 따라서 모듈 스코프에 cache + 동시 fetch 시 inflight Promise 공유로 충분.
//
// 사용:
//   const r = await postWithCsrf('/api/analyze', payload);
// 403 응답이 오면 (e.g. 백엔드 재시작으로 세션 폐기) 토큰 1회 강제 갱신 후 재시도.
//
// 헤더 이름은 백엔드 services/api/analyze.py 의 request.headers.get("X-CSRF-Token")
// 와 정확히 일치해야 함.

let cached: string | null = null;
let inflight: Promise<string> | null = null;

export async function getCsrfToken(forceRefresh = false): Promise<string> {
  if (!forceRefresh && cached) return cached;
  if (inflight) return inflight;
  inflight = fetch('/api/csrf', { credentials: 'include' })
    .then(r => {
      if (!r.ok) throw new Error(`GET /api/csrf failed: ${r.status}`);
      return r.json();
    })
    .then((j: { token: string }) => {
      cached = j.token;
      return j.token;
    })
    .finally(() => {
      inflight = null;
    });
  return inflight;
}

export async function postWithCsrf(url: string, body: unknown): Promise<Response> {
  const send = (token: string) =>
    fetch(url, {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRF-Token': token,
      },
      body: JSON.stringify(body),
    });

  let token = await getCsrfToken();
  let resp = await send(token);
  if (resp.status === 403) {
    // 세션 회전 후 stale token 가능성 — 1회 강제 refresh 후 재시도.
    token = await getCsrfToken(true);
    resp = await send(token);
  }
  return resp;
}
