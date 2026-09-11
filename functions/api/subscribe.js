/**
 * Cloudflare Pages Function: same-origin newsletter subscribe.
 *
 * The homepage form posts here instead of navigating to Buttondown's hosted
 * embed page, so the reader stays on the site and sees an inline status
 * message. Server-side we call Buttondown's REST API, which preserves double
 * opt-in: subscribers are created `unactivated` and get a confirmation email.
 *
 * Requires the BUTTONDOWN_API_KEY environment variable (Pages project
 * settings). Without it, this returns 503 and the frontend falls back to the
 * plain embed-form flow.
 */

const BUTTONDOWN_API = "https://api.buttondown.com/v1/subscribers";

function json(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export async function onRequestPost(context) {
  const { request, env } = context;

  if (!env.BUTTONDOWN_API_KEY) {
    return json(503, { ok: false, code: "unconfigured" });
  }

  let email = "";
  const contentType = request.headers.get("Content-Type") || "";
  try {
    if (contentType.includes("application/json")) {
      const body = await request.json();
      email = String(body.email || "");
    } else {
      const body = await request.formData();
      email = String(body.get("email") || "");
    }
  } catch {
    return json(400, { ok: false, code: "bad-request" });
  }

  email = email.trim();
  // Light-touch validation; Buttondown does the real thing.
  if (!email || email.length > 254 || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return json(400, { ok: false, code: "invalid-email" });
  }

  const upstream = await fetch(BUTTONDOWN_API, {
    method: "POST",
    headers: {
      Authorization: `Token ${env.BUTTONDOWN_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      email_address: email,
      // type defaults to "unactivated": double opt-in preserved.
      ip_address: request.headers.get("CF-Connecting-IP") || undefined,
    }),
  });

  if (upstream.status === 201) {
    return json(200, { ok: true });
  }

  if (upstream.status === 400) {
    // Duplicates masquerade as 400s; other 400s are addresses Buttondown
    // genuinely rejects and must not be reported as success.
    let detail = "";
    try {
      detail = JSON.stringify(await upstream.json());
    } catch {
      /* opaque body; fine */
    }
    if (/already|exist|duplicate/i.test(detail)) {
      return json(200, { ok: true, code: "already-subscribed" });
    }
    return json(400, { ok: false, code: "invalid-email" });
  }

  if (upstream.status === 429) {
    return json(429, { ok: false, code: "rate-limited" });
  }

  return json(502, { ok: false, code: "upstream-error" });
}
