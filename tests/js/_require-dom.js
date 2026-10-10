// "A skipped security test is not a passing one" — enforced.
//
// Several suites here need a DOM and skip cleanly when jsdom is absent, so a
// fresh clone stays green. That is the right default for a human, and the
// wrong one for CI: the workflow ran `node --test` with no install step, so
// 60 tests — every Job Board render-path assertion, the modal focus traps,
// the whole sanitizer suite — reported as skips inside a green run. "333
// passing" was a number that only existed on the maintainer's box.
//
// Set CI_REQUIRE_DOM=1 (see package.json's `test:ci`) and a missing jsdom
// becomes a hard failure instead. The skip reason is kept verbatim in the
// message so the fix is still one line away.

/**
 * Turn a skip reason into a failure when the environment demands a DOM.
 *
 * @param {string|false} reason - falsy when the suite can run.
 * @returns {string|false} the reason to pass to node:test's `skip` option.
 */
export function domSkip(reason) {
  if (!reason) return false;
  if (process.env.CI_REQUIRE_DOM === '1') {
    throw new Error(
      `CI_REQUIRE_DOM=1 but this suite cannot run: ${reason}\n`
      + 'Install the dev dependency (npm ci in frontend/) so these tests '
      + 'actually execute. A skipped test is not a passing one.',
    );
  }
  return reason;
}
