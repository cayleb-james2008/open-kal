// Headless probe of a running engine+dashboard: exact pageerror, console errors, failed requests,
// and whether #root actually renders. Usage: node dashboard/scripts/probe-live.mjs [url]
import fs from "node:fs";
import { chromium } from "playwright-core";

const base = process.argv[2] ?? process.env.OPENKAL_URL ?? "http://127.0.0.1:8787/";
const exe = [process.env.CHROME_PATH, "/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome"]
  .find((p) => p && fs.existsSync(p));
const VIEWPORTS = [{ name: "1440x900", width: 1440, height: 900 }, { name: "390x844", width: 390, height: 844 }];

const browser = await chromium.launch({ headless: true, executablePath: exe, args: ["--no-sandbox", "--disable-dev-shm-usage"] });
const origin = base.replace(/\/$/, "");
let failures = 0;
const report = [];

for (const vp of VIEWPORTS) {
  const ctx = await browser.newContext({ viewport: { width: vp.width, height: vp.height } });
  const page = await ctx.newPage();
  const pageErrors = [], consoleErrors = [], requestFailed = [], badResponses = [];
  page.on("pageerror", (e) => pageErrors.push(String(e?.stack ?? e)));
  page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
  page.on("requestfailed", (r) => requestFailed.push(`${r.method()} ${r.url()} ${r.failure()?.errorText ?? ""}`));
  page.on("response", (r) => { if (r.url().startsWith(origin) && r.status() >= 400) badResponses.push(`${r.status()} ${r.url()}`); });
  await page.goto(base, { waitUntil: "load" });
  await page.waitForTimeout(4500);
  const dom = await page.evaluate(() => {
    const root = document.querySelector("#root");
    const text = (root?.textContent ?? "").replace(/\s+/g, " ").trim();
    return {
      rootHtmlLength: root ? root.innerHTML.length : -1,
      rootChildCount: root ? root.children.length : -1,
      textSample: text.slice(0, 420),
      lanes: document.querySelectorAll('[id^="lane-"]').length,
      readinessRows: document.querySelectorAll("#arming li").length,
      tapeRows: document.querySelectorAll("#tape-log tbody tr").length,
      equityShown: (text.match(/\$[\d,]+\.\d{2}/g) ?? []).slice(0, 4),
      errorBanner: document.querySelector("[data-openkal-error]")?.textContent?.trim().slice(0, 400) ?? null,
      bank: (text.match(/\b(bid|ask|fair|spot)\b/g) ?? []).length,
    };
  });
  const api = await page.evaluate(async () => {
    try { const r = await fetch("/api/state", { cache: "no-store" }); const j = await r.json();
      return { ok: r.ok, live_armed: j?.live_armed, mode: j?.mode, slots: j?.slots?.length, equity: j?.account?.equity }; }
    catch (e) { return { error: String(e) }; }
  });
  const failedChecks = [];
  if (pageErrors.length) failedChecks.push(`${pageErrors.length} pageerror`);
  if (consoleErrors.length) failedChecks.push(`${consoleErrors.length} console error`);
  if (requestFailed.length) failedChecks.push(`${requestFailed.length} failed request`);
  if (badResponses.length) failedChecks.push(`${badResponses.length} 4xx/5xx`);
  if (dom.rootHtmlLength < 200) failedChecks.push(`#root blank (${dom.rootHtmlLength} chars)`);
  if (dom.lanes < 4) failedChecks.push(`lanes ${dom.lanes} < 4`);
  if (dom.readinessRows < 5) failedChecks.push(`readiness rows ${dom.readinessRows} < 5`);
  if (!dom.equityShown.length) failedChecks.push("no equity figure");
  if (failedChecks.length) failures++;
  report.push({ viewport: vp.name, failedChecks, pageErrors, consoleErrors, requestFailed, badResponses, dom, api });
  await ctx.close();
}
await browser.close();
console.log(JSON.stringify(report, null, 2));
console.log(failures ? `\nPROBE FAILED for ${failures}/${VIEWPORTS.length} viewports` : `\nPROBE OK for all ${VIEWPORTS.length} viewports`);
process.exit(failures ? 1 : 0);
