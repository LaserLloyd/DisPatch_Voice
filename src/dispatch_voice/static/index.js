// DisPatch Voice: the extension entry DisPatch's loader imports from
// /ext/voice/index.js?v=<assets hash>.
// Owner: track T7. Contracts: C-§A (JS entry and handshake), C-§E.1 (the
// frozen `shell` object and pane ownership), SPEC §3.5–§3.6.
//
// mount(shell, { assetsVersion }) wires the whole client side of voice chat
// into DisPatch through the shell's slots — nothing else in DisPatch knows
// voice exists:
//   chatHeader   the microphone button (hidden while voice is off, and on a
//                locked device for a bot it may not use — the `safe` ones it can
//                chat with keep it since 1.3.3)
//   settings     the Settings → Voice pane (replaces DisPatch's "add-on
//                absent" page while mounted; the loader puts it back on unmount)
//   bots         the Voice pill + popover on each Bots-tab card
//   tools        the three voice tools (Voice cloning, Built-in voices, Voice lab)
//   onLocked     a locked device closes any open voice chat at once
// Everything main.js did for voice before the split (probe the status, follow
// the switch, keep the admin roster for per-bot voices) lives here.
//
// Every sibling module is imported with the same `?v=` the entry was loaded
// with, so a cold open on cellular is one round of immutable fetches
// (feasibility M7), and every file is ONE module instance: the siblings
// import each other through their own `import.meta.url` query, which is this one.

export const EXT = { id: 'voice', api: [1, 1], version: '1.3.3' };  // = dispatch_voice.__version__ (pinned by tests/host/test_setup.py)

/** The query every sibling is imported with: the server's assets hash. */
export function versionQuery(opts, selfUrl = import.meta.url) {
  const v = opts && typeof opts.assetsVersion === 'string' && opts.assetsVersion;
  if (v) return `?v=${encodeURIComponent(v)}`;
  return new URL(selfUrl).search;
}

/** Load the client modules (exported for tests: the same graph mount() uses). */
export async function loadModules(q = '') {
  const [shellMod, drive, settings, tools] = await Promise.all([
    import(`./shell.js${q}`), import(`./drive.js${q}`),
    import(`./drive-settings.js${q}`), import(`./voice-tools.js${q}`),
  ]);
  return { shellMod, drive, settings, tools };
}

/**
 * Mount voice chat into DisPatch. Resolves to the handle the loader keeps:
 * { unmount(), onLocale() }. A failing /api/voice/status does NOT reject —
 * the pane then says it could not load and offers Try again; only a broken
 * install (a module that fails to import) rejects, and the loader shows its
 * own page for that.
 */
export async function mount(shell, opts = {}) {
  if (!shell || shell.EXT_API == null || shell.EXT_API < EXT.api[0] || shell.EXT_API > EXT.api[1]) {
    throw new Error(`DisPatch Voice needs extension API ${EXT.api[0]}–${EXT.api[1]}; this DisPatch offers ${shell && shell.EXT_API}.`);
  }
  const mods = await loadModules(versionQuery(opts));
  const app = createVoiceApp(shell, mods, { locked: !!(opts && opts.locked) });
  await app.start();
  return { unmount: () => app.stop(), onLocale: () => app.onLocale() };
}

/** The wiring, separate from mount() so tests can drive it with a fake shell. */
/** Where the keyboard focus is inside `root`: the nearest id plus the index of the
 *  focused control under it (rebuilt pages keep their ids). */
function focusPath(root) {
  const a = root && root.ownerDocument && root.ownerDocument.activeElement;
  if (!a || a === root || !root.contains(a)) return null;
  const anchor = a.id ? a : a.parentElement && a.parentElement.closest('[id]');
  if (!anchor || !root.contains(anchor)) return null;
  const all = anchor === a ? [a] : [...anchor.querySelectorAll('button, input, select, textarea, summary, [tabindex]')];
  return { id: anchor.id, i: Math.max(0, all.indexOf(a)), self: anchor === a };
}
function restoreFocus(root, p) {
  if (!p || !root) return;
  const doc = root.ownerDocument;
  const anchor = doc.getElementById(p.id);
  if (!anchor || !root.contains(anchor)) return;
  const n = p.self ? anchor : anchor.querySelectorAll('button, input, select, textarea, summary, [tabindex]')[p.i];
  if (n && typeof n.focus === 'function') { try { n.focus({ preventScroll: true }); } catch { /* detached */ } }
}

export function createVoiceApp(shell, { shellMod, drive: D, settings: DS, tools: VT }, { locked: lockedMount = false } = {}) {
  // A locked mount (1.3.3, DisPatch ≥ 2.3.4: `mount(shell, { locked: true })`):
  // the header button for the bots this device can chat with and the call
  // screen with this device's own settings. No Settings pane, no Bots-tab pill,
  // no tools: those change the server, and every /api/voice route is
  // unlocked-only. Its status is the add-on's `locked_status` row.
  const prevShell = shellMod.bindShell(shell);
  const { t, el, railIcon } = shellMod;
  const toast = (m, err) => { if (shell.toast) shell.toast(m, err); };
  const disposers = [];
  let status = null;          // the last /api/voice/status (null = not loaded or failed)
  let loading = false;
  let voiceOn = false;
  let roster = [];            // /api/bots/all: carries `voice` and `voice_rp`
  let cloningTool = false;    // tools.yaml has the Voice cloning tool on
  let paneRoot = null;
  let toolOpen = null;        // { id, close, root }
  let stopped = false;
  let statusSeq = 0;

  // ---- the microphone button in the chat header ---------------------------
  const button = el('button', {
    type: 'button', class: 'icon-btn ghost drive-btn', id: 'drive-btn', hidden: '',
    title: t('drive.button'), 'aria-label': t('drive.button_aria'),
  }, [railIcon(DS.MIC_ICON)]);
  const removeButton = shell.chatHeader && shell.chatHeader.addAction
    ? shell.chatHeader.addAction(button, { before: 'thread-menu-btn' }) : null;
  if (typeof removeButton === 'function') disposers.push(removeButton);
  else disposers.push(() => button.remove());

  const getBot = (id) => (lockedMount ? (shell.botById ? shell.botById(id) : null) : null) || roster.find((b) => b && b.id === id && typeof b.voice === 'string')
    || (shell.botById ? shell.botById(id) : null) || null;

  const drive = D.initDrive(button, {
    // The quick sheet re-reads the roster when it opens (review L6).
    reloadBots: lockedMount ? null : () => loadRoster(),
    getThread: () => (voiceOn && shell.activeThread ? shell.activeThread() : null),
    getBot,
    isLocked: () => !!(shell.isLocked && shell.isLocked()),
    // Voice switched off server-wide: the button hides and, if anyone reads it,
    // says THAT — not "unlock DisPatch" (V3 #6).
    isVoiceOff: () => !voiceOn,
    toast,
    // The thread's own face (or a letter when No-Image Mode is on).
    getAvatar: (thread, bot) => (shell.threadFace ? shell.threadFace(thread, bot) : { src: '', letter: '' }) || {},
    voiceStatus: () => status,
    openDriveSettings: lockedMount ? null : () => openSettings(),
    setBotVoice: lockedMount ? null : async (botId, v) => {
      const bots = await DS.setBotVoice(botId, v);
      if (bots.length) roster = bots;
      return bots;
    },
    history: shell.history,
  });
  disposers.push(() => drive.dispose());

  // ---- Settings → Voice ---------------------------------------------------
  if (!lockedMount && shell.settings && shell.settings.pane) {
    const dispose = shell.settings.pane('voice', {
      render(root) { paneRoot = root; renderPane(); },
      onShow() { refreshStatus(); },
      onHide() {},
    });
    if (typeof dispose === 'function') disposers.push(dispose);
  }

  function openSettings() {
    if (shell.settings && shell.settings.open) shell.settings.open('voice');
  }

  function cloningReady() {
    const p = status && status.plugins && status.plugins.cloning;
    return !!(cloningTool && p && p.ready);
  }

  function renderPane() {
    if (!paneRoot || stopped) return;
    // The page rebuilds from scratch; keep the keyboard where it was.
    const focus = focusPath(paneRoot);
    DS.mountDriveSettings(paneRoot, {
      status, loading,
      bots: () => roster,
      setEnabled,
      cloning: cloningReady(),
      openCloning: () => {
        if (shell.settings && typeof shell.settings.close === 'function') shell.settings.close();
        openTool('voice-cloning');
      },
      retry: () => refreshStatus({ force: true }),
      // A saved row: re-read the status quietly (rebuilds only if the server's
      // view differs from what the page merged in).
      saved: () => { refreshStatus(); },
      toast,
    });
    restoreFocus(paneRoot, focus);
  }

  async function setEnabled(on) {
    try {
      await DS.patchVoiceSettings({ enabled: on });
    } catch (e) {
      toast(e && e.status === 409 ? t('drive.switch_forced_error') : t('drive.err_save'), true);
      return false;
    }
    toast(on ? t('drive.switched_on') : t('drive.switched_off'));
    if (!on) drive.close({ quiet: true });
    await refreshStatus({ force: true });
    return true;
  }

  // ---- status -------------------------------------------------------------
  async function loadRoster() {
    try {
      const r = await DS.voiceJson('/api/bots/all');
      const bots = Array.isArray(r) ? r : (r && r.bots) || [];
      if (bots.length) roster = bots;
    } catch {
      try { const b = shell.allBots ? await shell.allBots() : []; if (Array.isArray(b) && b.length) roster = b; } catch { /* keep */ }
    }
  }

  async function loadToolFlag() {
    try {
      const r = await DS.voiceJson('/api/tools');
      const rows = Array.isArray(r) ? r : (r && (r.tools || r.rows)) || [];
      const row = rows.find((x) => x && x.id === 'voice-cloning');
      cloningTool = !!(row && row.enabled === true);
    } catch { cloningTool = false; }
  }

  /** Re-read /api/voice/status; rebuild what depends on it. */
  async function refreshStatus({ force = false } = {}) {
    if (stopped) return;
    if (lockedMount) { let st = null; try { st = await DS.lockedVoiceStatus(); } catch { st = null; } if (!stopped) apply(st); return; }
    if (shell.isLocked && shell.isLocked()) { apply(null); return; }
    const mine = ++statusSeq;
    if (!status) { loading = true; renderPane(); }
    let st = null;
    try { st = await DS.voiceStatus(); } catch { st = null; }
    if (stopped || mine !== statusSeq) return;
    loading = false;
    if (!force && st && status) {
      try { if (JSON.stringify(st) === JSON.stringify(status)) { renderPaneIfLoading(); return; } } catch { /* rebuild */ }
    }
    // A forced refresh re-reads the roster too: a cancelled Bots-tab change or
    // another device's edit must not leave the quick sheet stale (review L6).
    await Promise.all([st && st.enabled && (force || !roster.length) ? loadRoster() : null, loadToolFlag()]);
    if (stopped || mine !== statusSeq) return;
    apply(st);
  }
  function renderPaneIfLoading() { if (paneRoot && paneRoot.querySelector('.vtool-loading')) renderPane(); }

  let prefetchTimer = null;
  function apply(st) {
    const was = voiceOn;
    status = st;
    voiceOn = !!(st && st.enabled) && (lockedMount || !(shell.isLocked && shell.isLocked()));
    renderPane();
    paintPills();
    if (was && !voiceOn) drive.close({ quiet: true });
    drive.refresh();
    // Voice can run here: warm the worklet modules in the background (once).
    if (voiceOn && st.ready !== false && !prefetchTimer && typeof D.prefetchAudioWorklets === 'function') {
      prefetchTimer = setTimeout(() => { prefetchTimer = true; if (!stopped) D.prefetchAudioWorklets(); }, 1500);
    }
  }

  // ---- Bots tab: the Voice pill + popover ---------------------------------
  // The shell renders the roster on open and on register/dispose only, never
  // on a tab switch, so a pill cannot appear by being asked again. Every
  // non-Safe bot gets a host span at render time; the add-on fills or hides
  // it whenever the status changes (review M1: switching voice on in the
  // same Settings session shows the pills on the Bots tab at once).
  const pillHosts = new Set();  // { host, bot, onChange }
  function paintPill(h) {
    const pill = h.host.firstElementChild;
    if (pill && pill.querySelector('[aria-expanded="true"]')) return;  // an open popover stays
    if (pill && typeof pill.closePopover === 'function') pill.closePopover();
    if (!voiceOn || !status) { h.host.hidden = true; h.host.replaceChildren(); return; }
    // The shell's row copy carries unsaved edits; our roster the saved values.
    const mine = roster.find((b) => b && b.id === h.bot.id) || {};
    const view = { ...h.bot, ...h.edits };
    if (typeof view.voice !== 'string' && typeof mine.voice === 'string') view.voice = mine.voice;
    if (typeof view.voice_rp !== 'string' && typeof mine.voice_rp === 'string') view.voice_rp = mine.voice_rp;
    h.host.replaceChildren(DS.botVoicePill(view, status, (patch) => {
      Object.assign(h.edits, patch);   // until Save, only the shell and this row know
      h.onChange(patch);
    }, { notify: (m) => toast(m, true) }));
    h.host.hidden = false;
  }
  function paintPills() {
    for (const h of pillHosts) {
      if (!h.host.isConnected) { pillHosts.delete(h); continue; }
      paintPill(h);
    }
  }
  if (!lockedMount && shell.bots && shell.bots.rowExtra) {
    const dispose = shell.bots.rowExtra((bot, onChange) => {
      if (!bot) return null;
      // Rows from an earlier render are gone: forget them.
      for (const h of pillHosts) if (!h.host.isConnected) pillHosts.delete(h);
      const h = { host: el('span', { class: 'bm-voice-host' }), bot, onChange, edits: {} };
      pillHosts.add(h);
      paintPill(h);
      return h.host;
    });
    if (typeof dispose === 'function') disposers.push(() => { pillHosts.clear(); dispose(); });
  }

  // ---- tools --------------------------------------------------------------
  function toolCtx() {
    return {
      toast,
      confirm: (msg, o) => (shell.confirm ? shell.confirm(msg, o) : Promise.resolve(false)),
      prompt: (msg, def) => (shell.prompt ? shell.prompt(msg, def) : Promise.resolve(null)),
      onChanged: () => refreshStatus({ force: true }),
      openVoiceSettings: () => { closeTool(); openSettings(); },
    };
  }

  function closeTool() {
    if (!toolOpen) return;
    const { close, observer } = toolOpen;
    toolOpen = null;
    if (observer) observer.disconnect();
    VT.unmountVoiceTool();
    if (typeof close === 'function') { try { close(); } catch { /* the shell's */ } }
  }

  function openTool(id) {
    if (stopped || (shell.isLocked && shell.isLocked())) return;
    const tile = VT.VOICE_TOOL_TILES.find((x) => x.id === id);
    if (!tile || !shell.tools || !shell.tools.openPane) return;
    closeTool();
    const me = { id, close: null, observer: null };
    toolOpen = me;
    me.close = shell.tools.openPane({
      id, title: t(tile.nameKey), subtitle: t(tile.subKey), iconPaths: tile.iconPaths,
      render(root) {
        VT.mountVoiceTool(root, id, toolCtx());
        // The shell owns the pane: when it empties or drops it, stop what the
        // tool has running (a recording, audio, a poll).
        if (globalThis.MutationObserver) {
          me.observer = new MutationObserver(() => {
            if (!root.isConnected || !root.firstChild) { if (toolOpen === me) { toolOpen = null; me.observer.disconnect(); VT.unmountVoiceTool(); } }
          });
          me.observer.observe(root, { childList: true });
        }
      },
      onClose() { if (toolOpen === me) { toolOpen = null; if (me.observer) me.observer.disconnect(); VT.unmountVoiceTool(); } },
    });
  }

  if (!lockedMount && shell.tools && shell.tools.register) {
    const dispose = shell.tools.register(VT.VOICE_TOOL_TILES.map((tile) => ({
      id: tile.id, feature: tile.feature, iconPaths: tile.iconPaths, nameKey: tile.nameKey, ariaKey: tile.ariaKey,
      open: (id) => openTool(id || tile.id),
    })));
    if (typeof dispose === 'function') disposers.push(dispose);
  }

  // ---- lock, thread changes ------------------------------------------------
  if (shell.onLocked) {
    const off = shell.onLocked(() => {
      drive.close({ quiet: true });
      closeTool();
      voiceOn = false;
      drive.refresh();
    });
    if (typeof off === 'function') disposers.push(off);
  }
  // Optional (Amendment T7a-2): a shell that reports thread changes lets the
  // button hide for a Safe-Mode bot's thread before it is tapped.
  if (typeof shell.onThread === 'function') {
    const off = shell.onThread(() => drive.refresh());
    if (typeof off === 'function') disposers.push(off);
  }

  return {
    async start() { await refreshStatus({ force: true }); },
    stop() {
      if (stopped) return;
      stopped = true;
      if (prefetchTimer && prefetchTimer !== true) clearTimeout(prefetchTimer);
      closeTool();
      for (const fn of disposers.splice(0).reverse()) { try { fn(); } catch { /* keep going */ } }
      paneRoot = null;
      if (shellMod.getShell() === shell) shellMod.bindShell(prevShell);
    },
    onLocale() {
      if (stopped) return;
      button.title = t('drive.button');
      renderPane();
      drive.refresh();
      if (toolOpen) { const id = toolOpen.id; openTool(id); }
    },
    // test hooks
    get status() { return status; },
    get voiceOn() { return voiceOn; },
    get button() { return button; },
    refreshStatus,
    openTool,
    closeTool,
  };
}
