// Package bridge implements the wa-bridge stdio protocol, version 2
// (docs/whatsapp/bridge-protocol.md) on top of an abstract WhatsApp Client.
package bridge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Exit codes (protocol §1).
const (
	ExitOK      = 0
	ExitError   = 1
	ExitBadArgs = 2
	ExitBudget  = 3
	ExitLocked  = 75
)

const (
	netTimeout      = 25 * time.Second // protocol §2: never wait longer on one request
	downloadTimeout = 2 * time.Minute  // inbound media, outside the request path
	editWindow      = 15 * time.Minute // protocol §3 edit
	maxMedia        = 100 << 20        // protocol §3 send_media, also the inbound cap
	maxQuotedMedia  = 25 << 20         // protocol §4 quoted.media, on bytes read
	maxContextText  = 4000             // protocol §4 context.text, in characters
	maxLinks        = 10               // protocol §4 quoted.links
	maxMentions     = 5                // protocol §3.1
	maxHistory      = 50               // protocol §3 history count
	historyEvery    = time.Hour        // at most one history request per chat
	historyWait     = 60 * time.Second // an open request expires
	membersTTL      = 10 * time.Minute // members cache, also cleared on chat changes
	editReactCap    = 60               // fixed per-minute cap for edit + react
	drainWait       = 1 * time.Second  // in-flight wait at exit (EOF must exit < 2 s)
)

// Opener opens the store and returns a client for it. It is called only after
// the store lock is held.
type Opener func(cfg Config, log *Logger) (Client, error)

// Options injects the environment; zero values pick production defaults.
type Options struct {
	Open Opener
	Now  func() time.Time
}

// Run is the whole process: it returns the exit code.
func Run(args []string, stdin io.Reader, stdout, stderr io.Writer, opt Options) (code int) {
	log := NewLogger(stderr)
	defer func() {
		if r := recover(); r != nil {
			log.Printf("panic: %T", r)
			code = ExitError
		}
	}()
	cfg, err := ParseFlags(args, stderr)
	if err != nil {
		log.Printf("bad flags: %v", err)
		return ExitBadArgs
	}
	// Startup order step 2: lock before the store is opened.
	lock, err := lockStore(cfg.Store)
	if errors.Is(err, errLocked) {
		log.Printf("store locked by another bridge; exiting 75")
		return ExitLocked
	}
	if err != nil {
		log.Printf("lock failed: %v", err)
		return ExitError
	}
	defer lock.Close()
	if err := os.MkdirAll(cfg.MediaDir, 0o700); err != nil {
		log.Printf("media dir: %v", err)
		return ExitError
	}
	_ = os.Chmod(cfg.MediaDir, 0o700)
	if opt.Open == nil {
		log.Printf("no client opener")
		return ExitError
	}
	if opt.Now == nil {
		opt.Now = time.Now
	}
	// Step 3: open the store.
	client, err := opt.Open(cfg, log)
	if err != nil {
		log.Printf("store open failed: %v", err)
		return ExitError
	}
	b := newBridge(cfg, client, &writer{w: stdout}, log, opt.Now)
	defer client.Close()
	return b.run(stdin)
}

// Bridge is the protocol state machine.
type Bridge struct {
	cfg    Config
	client Client
	w      *writer
	log    *Logger
	now    func() time.Time

	exit     chan int
	inflight sync.WaitGroup
	closing  bool

	mu        sync.Mutex
	helloDone bool
	allow     map[string]bool
	context   map[string]bool // ⊆ allow
	deny      map[string]bool
	trusted   map[int]bool
	members   map[string]membersEntry
	histOpen  map[string]time.Time // chat -> deadline of the open history request
	histLast  map[string]time.Time // chat -> last history request
	used      int                  // counted outbound actions
	exhausted bool
	sendWin   window
	editWin   window
	dryN      int
	sent      *sentRing
	senders   *senderCache
	dropped   map[string]int
	available bool // desired presence; default unavailable
	pairing   bool
	syncing   bool
	identity  Identity
}

func newBridge(cfg Config, c Client, w *writer, log *Logger, now func() time.Time) *Bridge {
	b := &Bridge{
		cfg: cfg, client: c, w: w, log: log, now: now,
		exit:     make(chan int, 1),
		allow:    set(cfg.Allow),
		context:  set(cfg.Context),
		deny:     set(cfg.Deny),
		members:  map[string]membersEntry{},
		histOpen: map[string]time.Time{},
		histLast: map[string]time.Time{},
		trusted:  map[int]bool{},
		sendWin:  window{limit: cfg.MaxPerMin},
		editWin:  window{limit: editReactCap},
		senders:  newSenderCache(),
		dropped:  map[string]int{},
	}
	for _, d := range cfg.TrustedDevices {
		b.trusted[d] = true
	}
	b.sent = loadSentRing(cfg.Store + ".sent.json")
	return b
}

func set(xs []string) map[string]bool {
	m := make(map[string]bool, len(xs))
	for _, x := range xs {
		m[x] = true
	}
	return m
}

func (b *Bridge) run(stdin io.Reader) int {
	b.client.SetHandler(b.onEvent)
	linked := b.client.Linked()
	if linked {
		b.mu.Lock()
		b.identity = b.client.Identity()
		b.mu.Unlock()
	}
	// Step 4.
	b.w.event("ready", map[string]any{"protocol": ProtocolVersion, "bridge_version": BridgeVersion, "linked": linked})
	b.log.Printf("ready linked=%v dry_run=%v test=%v allow=%d deny=%d budget=%d", linked, b.cfg.DryRun, b.cfg.Test, len(b.cfg.Allow), len(b.cfg.Deny), b.cfg.Budget)
	// Step 5.
	if linked {
		b.goSafe(func() {
			if err := b.client.Connect(); err != nil {
				b.log.Printf("connect failed: %v", err)
				b.w.event("offline", map[string]any{"reason": "connect_failed"})
			}
		})
	}

	lines := make(chan []byte)
	readErr := make(chan error, 1)
	go func() {
		lr := newLineReader(stdin)
		for {
			line, err := lr.next()
			if errors.Is(err, errTooLong) {
				lines <- nil
				continue
			}
			if err != nil {
				readErr <- err
				return
			}
			lines <- line
		}
	}()

	for {
		// A requested exit (budget, version mismatch, worker panic) wins over
		// any further input, so no command runs after the fatal response.
		select {
		case code := <-b.exit:
			return b.shutdown(code)
		default:
		}
		select {
		case line := <-lines:
			if line == nil {
				b.w.fail(nil, "bad_frame", "line over 1 MiB")
				b.log.Printf("bad_frame: line over 1 MiB")
				continue
			}
			if len(strings.TrimSpace(string(line))) == 0 {
				continue
			}
			b.handleLine(line)
		case err := <-readErr:
			if !errors.Is(err, io.EOF) {
				b.log.Printf("stdin error: %v", err)
			}
			b.log.Printf("stdin closed; exiting 0")
			return b.shutdown(ExitOK)
		case code := <-b.exit:
			return b.shutdown(code)
		}
	}
}

func (b *Bridge) shutdown(code int) int {
	b.mu.Lock()
	b.closing = true
	b.mu.Unlock()
	done := make(chan struct{})
	go func() { b.inflight.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(drainWait):
		b.log.Printf("exit with requests in flight")
	}
	b.client.Disconnect()
	b.log.Printf("exit %d", code)
	return code
}

// requestExit asks the read loop to stop with code; the first request wins.
func (b *Bridge) requestExit(code int) {
	select {
	case b.exit <- code:
	default:
	}
}

// goSafe runs f in a goroutine; a panic logs its type and exits 1.
func (b *Bridge) goSafe(f func()) {
	go func() {
		defer func() {
			if r := recover(); r != nil {
				b.log.Printf("panic in worker: %T", r)
				b.requestExit(ExitError)
			}
		}()
		f()
	}()
}

// async runs a network op off the read loop, tracked for the exit drain.
func (b *Bridge) async(f func(ctx context.Context)) {
	b.inflight.Add(1)
	b.goSafe(func() {
		defer b.inflight.Done()
		ctx, cancel := context.WithTimeout(context.Background(), netTimeout)
		defer cancel()
		f(ctx)
	})
}

// request is the union of every command's arguments.
type request struct {
	Op       string    `json:"op"`
	Protocol string    `json:"protocol"`
	Chat     string    `json:"chat"`
	Text     *string   `json:"text"`
	Msg      string    `json:"msg"`
	FromMe   *bool     `json:"from_me"`
	Emoji    *string   `json:"emoji"`
	Path     string    `json:"path"`
	Kind     string    `json:"kind"`
	Caption  string    `json:"caption"`
	FileName string    `json:"file_name"`
	Mime     string    `json:"mime"`
	State    string    `json:"state"`
	Allow    *[]string `json:"allow"`
	Context  *[]string `json:"context"`
	Mentions []string  `json:"mentions"`
	Before   string    `json:"before_msg"`
	Count    int       `json:"count"`
	ReplyTo  *struct {
		Msg    string `json:"msg"`
		FromMe bool   `json:"from_me"`
	} `json:"reply_to"`
}

func (b *Bridge) handleLine(line []byte) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(line, &raw); err != nil || raw == nil {
		b.w.fail(nil, "bad_frame", "invalid JSON object")
		b.log.Printf("bad_frame: invalid JSON (len=%d)", len(line))
		return
	}
	idRaw, ok := raw["id"]
	if !ok {
		b.w.fail(nil, "bad_frame", "missing id")
		b.log.Printf("bad_frame: missing id")
		return
	}
	id, err := strconv.ParseInt(strings.TrimSpace(string(idRaw)), 10, 64)
	if err != nil {
		b.w.fail(nil, "bad_frame", "id must be an integer")
		b.log.Printf("bad_frame: bad id")
		return
	}
	var req request
	if err := json.Unmarshal(line, &req); err != nil {
		b.fail(id, "bad_args", "argument has the wrong type")
		return
	}
	b.log.Printf("req id=%d op=%s", id, logToken(req.Op))
	b.dispatch(id, &req)
}

// logToken keeps op names loggable without echoing arbitrary input.
var opRe = regexp.MustCompile(`^[a-z_]{1,32}$`)

func logToken(s string) string {
	if opRe.MatchString(s) {
		return s
	}
	return fmt.Sprintf("<%d bytes>", len(s))
}

func (b *Bridge) fail(id int64, code, msg string) {
	b.w.fail(&id, code, msg)
	b.log.Printf("resp id=%d err=%s", id, code)
}

func (b *Bridge) ok(id int64, data any) {
	b.w.ok(id, data)
	b.log.Printf("resp id=%d ok", id)
}

func (b *Bridge) dispatch(id int64, r *request) {
	if r.Op == "hello" {
		b.hello(id, r)
		return
	}
	b.mu.Lock()
	hello := b.helloDone
	b.mu.Unlock()
	if !hello {
		if _, known := ops[r.Op]; known {
			b.fail(id, "bad_args", "hello must be the first command")
		} else {
			b.fail(id, "unknown_op", "unknown op")
		}
		return
	}
	h, ok := ops[r.Op]
	if !ok {
		b.fail(id, "unknown_op", "unknown op")
		return
	}
	h(b, id, r)
}

var ops map[string]func(*Bridge, int64, *request)

func init() {
	ops = map[string]func(*Bridge, int64, *request){
		"pair":          (*Bridge).pair,
		"list_chats":    (*Bridge).listChats,
		"send_text":     (*Bridge).sendText,
		"edit":          (*Bridge).edit,
		"react":         (*Bridge).react,
		"send_media":    (*Bridge).sendMedia,
		"presence":      (*Bridge).presence,
		"chat_presence": (*Bridge).chatPresence,
		"set_allow":     (*Bridge).setAllow,
		"members":       (*Bridge).membersOp,
		"history":       (*Bridge).history,
		"status":        (*Bridge).status,
		"logout":        (*Bridge).logout,
	}
}

// ---- hello / status / set_allow -------------------------------------------

func (b *Bridge) hello(id int64, r *request) {
	if r.Protocol != ProtocolVersion {
		b.fail(id, "version_mismatch", "this bridge speaks protocol "+ProtocolVersion)
		b.requestExit(ExitBadArgs)
		return
	}
	b.mu.Lock()
	b.helloDone = true
	d := map[string]any{
		"protocol":       ProtocolVersion,
		"bridge_version": BridgeVersion,
		"linked":         b.client.Linked(),
		"allow":          sorted(b.allow),
		"context":        sorted(b.context),
		"deny":           append([]string{}, b.cfg.Deny...),
		"dry_run":        b.cfg.DryRun,
		"test":           b.cfg.Test,
	}
	if b.client.Linked() {
		ident := b.client.Identity()
		d["jid"], d["lid"], d["device"] = ident.JID, ident.LID, ident.Device
	}
	if b.cfg.HasBudget() {
		d["budget_left"] = b.cfg.Budget - b.used
	}
	b.mu.Unlock()
	b.ok(id, d)
}

func (b *Bridge) status(id int64, _ *request) {
	b.mu.Lock()
	now := b.now()
	dropped := make(map[string]int, len(b.dropped))
	for k, v := range b.dropped {
		dropped[k] = v
	}
	d := map[string]any{
		"linked":         b.client.Linked(),
		"connected":      b.client.Connected(),
		"sent_last_min":  b.sendWin.count(now),
		"total_outbound": b.used,
		"dropped":        dropped,
	}
	if b.cfg.HasBudget() {
		d["budget_left"] = b.cfg.Budget - b.used
	}
	b.mu.Unlock()
	b.ok(id, d)
}

func (b *Bridge) setAllow(id int64, r *request) {
	if b.cfg.Test {
		b.fail(id, "test_mode", "set_allow is disabled under --test")
		return
	}
	if r.Allow == nil {
		b.fail(id, "bad_args", "allow is required")
		return
	}
	next := map[string]bool{}
	for _, j := range *r.Allow {
		if !validJID(j) {
			b.fail(id, "bad_args", "allow holds an invalid JID")
			return
		}
		next[j] = true
	}
	b.mu.Lock()
	for j := range next {
		if b.deny[j] {
			b.mu.Unlock()
			b.fail(id, "denied", "a JID is on the never-list")
			return
		}
	}
	ctxSet := map[string]bool{}
	if r.Context != nil {
		for _, j := range *r.Context {
			if !next[j] {
				b.mu.Unlock()
				b.fail(id, "bad_args", "context must be a subset of allow")
				return
			}
			ctxSet[j] = true
		}
	} else {
		for j := range b.context { // omitted: keep the current set minus chats no longer allowed
			if next[j] {
				ctxSet[j] = true
			}
		}
	}
	b.allow, b.context = next, ctxSet
	out, ctxOut := sorted(b.allow), sorted(b.context)
	b.mu.Unlock()
	b.log.Printf("allow set: %d chats, %d context", len(out), len(ctxOut))
	b.ok(id, map[string]any{"allow": out, "context": ctxOut})
}

func sorted(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// ---- the outbound gate (protocol §3 rules) ---------------------------------

type bucket int

const (
	bucketSend bucket = iota // send_text, send_media: --max-per-min
	bucketEdit               // edit, react: fixed 60/min
	bucketNone               // history: budget only, no per-minute window
)

// gate runs checks 1-5 for a counted outbound action, with extra validation
// (not_found, edit_window, too_large) between check 2 and check 3 so that a
// malformed request never spends budget. On success it counts the action and
// returns "". Must be called from the read loop.
func (b *Bridge) gate(id int64, chat string, bk bucket, extra func() (string, string)) bool {
	b.mu.Lock()
	wasExhausted := b.exhausted
	code, msg := b.gateLocked(chat, bk, extra)
	if code == "" {
		now := b.now()
		b.used++
		switch bk {
		case bucketSend:
			b.sendWin.add(now)
		case bucketEdit:
			b.editWin.add(now)
		}
	}
	used := b.used
	b.mu.Unlock()
	if code == "" {
		return true
	}
	b.fail(id, code, msg)
	if code == "budget" && !wasExhausted {
		b.w.event("budget_exhausted", map[string]any{"used": used})
		b.log.Printf("budget exhausted used=%d; exiting 3", used)
		b.requestExit(ExitBudget)
	}
	return false
}

func (b *Bridge) gateLocked(chat string, bk bucket, extra func() (string, string)) (string, string) {
	if b.exhausted {
		return "budget", "the budget is spent"
	}
	if b.deny[chat] { // 1
		return "denied", "that chat is on the never-list"
	}
	if !b.allow[chat] { // 2
		return "not_allowed", "that chat is not allowlisted"
	}
	if extra != nil {
		if c, m := extra(); c != "" {
			return c, m
		}
	}
	now := b.now()
	if bk == bucketSend && b.sendWin.full(now) || bk == bucketEdit && b.editWin.full(now) { // 3
		return "rate", "the per-minute cap was reached"
	}
	if b.cfg.HasBudget() && b.used >= b.cfg.Budget { // 4
		b.exhausted = true
		return "budget", "the budget is spent"
	}
	return b.linkLocked() // 5
}

func (b *Bridge) linkLocked() (string, string) {
	if !b.client.Linked() {
		return "not_linked", "no device is linked"
	}
	if !b.client.Connected() {
		return "offline", "not connected to WhatsApp"
	}
	return "", ""
}

// dryID returns the next fake id under --dry-run.
func (b *Bridge) dryID(chat string) string {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.dryN++
	id := fmt.Sprintf("dry-%d", b.dryN)
	b.sent.push(sentEntry{ID: id, Chat: chat, At: b.now()}) // not persisted
	return id
}

func (b *Bridge) recordSent(id, chat string) {
	b.mu.Lock()
	b.sent.add(sentEntry{ID: id, Chat: chat, At: b.now()})
	b.mu.Unlock()
}

// netErr maps a client error to a response.
func (b *Bridge) netErr(id int64, op string, err error) {
	b.log.Printf("%s failed id=%d: %v", op, id, err)
	if errors.Is(err, context.DeadlineExceeded) {
		b.fail(id, "offline", "WhatsApp did not answer in time")
		return
	}
	b.fail(id, "internal", "the WhatsApp request failed")
}

// participant resolves the author of msg in chat for quotes and reactions.
func (b *Bridge) participant(chat, msg string, fromMe bool) (string, bool) {
	if fromMe {
		return "", true // "" means our own account
	}
	return b.senders.get(chat, msg)
}

// ---- outbound commands ------------------------------------------------------

func (b *Bridge) sendText(id int64, r *request) {
	if r.Chat == "" || r.Text == nil || *r.Text == "" || (r.ReplyTo != nil && r.ReplyTo.Msg == "") {
		b.fail(id, "bad_args", "chat and text are required")
		return
	}
	var reply *ReplyRef
	extra := func() (string, string) {
		if r.ReplyTo == nil {
			return "", ""
		}
		sender, ok := b.participant(r.Chat, r.ReplyTo.Msg, r.ReplyTo.FromMe)
		if !ok {
			return "not_found", "unknown reply_to message"
		}
		reply = &ReplyRef{MsgID: r.ReplyTo.Msg, Sender: sender}
		return "", ""
	}
	if !b.gate(id, r.Chat, bucketSend, extra) {
		return
	}
	chat, text, asked := r.Chat, *r.Text, r.Mentions
	if b.cfg.DryRun {
		msg := b.dryID(chat)
		kept := b.checkMentions(context.Background(), chat, text, asked, false)
		b.log.Printf("send_text dry id=%d msg=%s len=%d mentions=%d", id, msg, len(text), len(kept))
		b.ok(id, map[string]any{"msg": msg, "mentioned": kept})
		return
	}
	b.async(func(ctx context.Context) {
		kept := b.checkMentions(ctx, chat, text, asked, true)
		msg, err := b.client.SendText(ctx, chat, text, reply, kept)
		if err != nil {
			b.netErr(id, "send_text", err)
			return
		}
		b.recordSent(msg, chat)
		b.log.Printf("send_text id=%d msg=%s len=%d mentions=%d", id, msg, len(text), len(kept))
		b.ok(id, map[string]any{"msg": msg, "mentioned": kept})
	})
}

func (b *Bridge) edit(id int64, r *request) {
	if r.Chat == "" || r.Msg == "" || r.Text == nil || *r.Text == "" {
		b.fail(id, "bad_args", "chat, msg and text are required")
		return
	}
	extra := func() (string, string) {
		e, ok := b.sent.get(r.Msg)
		if !ok || e.Chat != r.Chat {
			return "not_found", "not a message this bridge sent"
		}
		if b.now().Sub(e.At) > editWindow {
			return "edit_window", "the message is older than 15 minutes"
		}
		return "", ""
	}
	if !b.gate(id, r.Chat, bucketEdit, extra) {
		return
	}
	chat, msg, text, asked := r.Chat, r.Msg, *r.Text, r.Mentions
	if b.cfg.DryRun {
		kept := b.checkMentions(context.Background(), chat, text, asked, false)
		b.log.Printf("edit dry id=%d msg=%s len=%d mentions=%d", id, msg, len(text), len(kept))
		b.ok(id, map[string]any{"mentioned": kept})
		return
	}
	b.async(func(ctx context.Context) {
		kept := b.checkMentions(ctx, chat, text, asked, true)
		if err := b.client.Edit(ctx, chat, msg, text, kept); err != nil {
			b.netErr(id, "edit", err)
			return
		}
		b.log.Printf("edit id=%d msg=%s len=%d mentions=%d", id, msg, len(text), len(kept))
		b.ok(id, map[string]any{"mentioned": kept})
	})
}

func (b *Bridge) react(id int64, r *request) {
	if r.Chat == "" || r.Msg == "" || r.Emoji == nil {
		b.fail(id, "bad_args", "chat, msg and emoji are required")
		return
	}
	fromMe := r.FromMe != nil && *r.FromMe
	var sender string
	extra := func() (string, string) {
		s, ok := b.participant(r.Chat, r.Msg, fromMe)
		if !ok {
			return "not_found", "unknown message for a reaction"
		}
		sender = s
		return "", ""
	}
	if !b.gate(id, r.Chat, bucketEdit, extra) {
		return
	}
	chat, msg, emoji := r.Chat, r.Msg, *r.Emoji
	if b.cfg.DryRun {
		b.log.Printf("react dry id=%d msg=%s", id, msg)
		b.ok(id, nil)
		return
	}
	b.async(func(ctx context.Context) {
		if err := b.client.React(ctx, chat, sender, msg, emoji); err != nil {
			b.netErr(id, "react", err)
			return
		}
		b.log.Printf("react id=%d msg=%s", id, msg)
		b.ok(id, nil)
	})
}

func (b *Bridge) sendMedia(id int64, r *request) {
	if r.Chat == "" || r.Path == "" || (r.Kind != "document" && r.Kind != "image") {
		b.fail(id, "bad_args", "chat, path and kind (document|image) are required")
		return
	}
	extra := func() (string, string) {
		st, err := os.Stat(r.Path)
		if err != nil || !st.Mode().IsRegular() {
			return "bad_args", "path is not a regular file"
		}
		if st.Size() > maxMedia {
			return "too_large", "the file is over 100 MB"
		}
		return "", ""
	}
	if !b.gate(id, r.Chat, bucketSend, extra) {
		return
	}
	m := MediaOut{Chat: r.Chat, Path: r.Path, Kind: r.Kind, Caption: r.Caption, FileName: r.FileName, Mime: r.Mime}
	if m.FileName == "" {
		m.FileName = filepath.Base(m.Path)
	}
	if m.Mime == "" {
		m.Mime = mime.TypeByExtension(strings.ToLower(filepath.Ext(m.FileName)))
		if m.Mime == "" {
			m.Mime = "application/octet-stream"
		}
	}
	if b.cfg.DryRun {
		msg := b.dryID(m.Chat)
		b.log.Printf("send_media dry id=%d msg=%s kind=%s", id, msg, m.Kind)
		b.ok(id, map[string]any{"msg": msg})
		return
	}
	b.async(func(ctx context.Context) {
		data, err := os.ReadFile(m.Path)
		if err != nil || len(data) > maxMedia {
			b.fail(id, "bad_args", "the file could not be read")
			return
		}
		m.Data = data
		msg, err := b.client.SendMedia(ctx, m)
		if err != nil {
			b.netErr(id, "send_media", err)
			return
		}
		b.recordSent(msg, m.Chat)
		b.log.Printf("send_media id=%d msg=%s kind=%s size=%d", id, msg, m.Kind, len(data))
		b.ok(id, map[string]any{"msg": msg})
	})
}

// ---- uncounted commands -----------------------------------------------------

func (b *Bridge) presence(id int64, r *request) {
	if r.State != "available" && r.State != "unavailable" {
		b.fail(id, "bad_args", "state must be available or unavailable")
		return
	}
	avail := r.State == "available"
	b.mu.Lock()
	b.available = avail
	b.mu.Unlock()
	// Remembered and re-applied on every connect; sent now if connected.
	if b.cfg.DryRun || !b.client.Connected() {
		b.ok(id, nil)
		return
	}
	b.async(func(ctx context.Context) {
		if err := b.client.SetPresence(ctx, avail); err != nil {
			b.netErr(id, "presence", err)
			return
		}
		b.ok(id, nil)
	})
}

func (b *Bridge) chatPresence(id int64, r *request) {
	if r.Chat == "" || (r.State != "composing" && r.State != "paused") {
		b.fail(id, "bad_args", "chat and state (composing|paused) are required")
		return
	}
	b.mu.Lock()
	var code, msg string
	switch {
	case b.deny[r.Chat]:
		code, msg = "denied", "that chat is on the never-list"
	case !b.allow[r.Chat]:
		code, msg = "not_allowed", "that chat is not allowlisted"
	default:
		code, msg = b.linkLocked()
	}
	b.mu.Unlock()
	if code != "" {
		b.fail(id, code, msg)
		return
	}
	if b.cfg.DryRun {
		b.ok(id, nil)
		return
	}
	chat, composing := r.Chat, r.State == "composing"
	b.async(func(ctx context.Context) {
		if err := b.client.ChatPresence(ctx, chat, composing); err != nil {
			b.netErr(id, "chat_presence", err)
			return
		}
		b.ok(id, nil)
	})
}

func (b *Bridge) listChats(id int64, _ *request) {
	b.mu.Lock()
	code, msg := b.linkLocked()
	b.mu.Unlock()
	if code != "" {
		b.fail(id, code, msg)
		return
	}
	b.async(func(ctx context.Context) {
		chats, err := b.client.ListChats(ctx)
		if err != nil {
			b.netErr(id, "list_chats", err)
			return
		}
		if chats == nil {
			chats = []ChatInfo{}
		}
		b.log.Printf("list_chats id=%d n=%d", id, len(chats)) // subjects never logged
		b.ok(id, map[string]any{"chats": chats})
	})
}

func (b *Bridge) pair(id int64, _ *request) {
	if b.cfg.Test {
		b.fail(id, "test_mode", "pair is disabled under --test")
		return
	}
	if b.client.Linked() {
		b.fail(id, "already_linked", "a device is already linked")
		return
	}
	b.mu.Lock()
	busy := b.pairing
	b.pairing = true
	b.mu.Unlock()
	if busy {
		b.ok(id, nil) // already pairing; codes keep flowing
		return
	}
	b.goSafe(func() {
		defer func() {
			b.mu.Lock()
			b.pairing = false
			b.mu.Unlock()
		}()
		ch, err := b.client.StartPair(context.Background())
		if err != nil {
			b.netErr(id, "pair", err)
			return
		}
		b.ok(id, nil)
		b.log.Printf("pairing started")
		for it := range ch {
			switch {
			case it.Code != "":
				b.w.event("qr", map[string]any{"code": it.Code, "expires_s": int(it.Timeout.Round(time.Second) / time.Second)})
				b.log.Printf("qr code emitted")
			case it.Done == "success":
				b.log.Printf("pairing succeeded")
				return
			case it.Done == "timeout":
				b.w.event("error", map[string]any{"code": "pair_timeout", "error": "the QR code was not scanned in time"})
				b.log.Printf("pairing timed out")
				return
			case it.Done != "":
				b.w.event("error", map[string]any{"code": "pair_failed", "error": "pairing failed"})
				b.log.Printf("pairing failed: %s", it.Err)
				return
			}
		}
	})
}

func (b *Bridge) logout(id int64, _ *request) {
	if b.cfg.Test {
		b.fail(id, "test_mode", "logout is disabled under --test")
		return
	}
	b.mu.Lock()
	code, msg := b.linkLocked()
	b.mu.Unlock()
	if code != "" {
		b.fail(id, code, msg)
		return
	}
	b.async(func(ctx context.Context) {
		if err := b.client.Logout(ctx); err != nil {
			b.netErr(id, "logout", err)
			return
		}
		b.mu.Lock()
		b.identity = Identity{}
		b.mu.Unlock()
		b.ok(id, nil)
		b.w.event("logged_out", map[string]any{"reason": "logout"})
		b.log.Printf("logged out by request; staying up unlinked")
	})
}

// ---- events from WhatsApp ---------------------------------------------------

func (b *Bridge) onEvent(e Event) {
	switch {
	case e.Connected:
		ident := b.client.Identity()
		b.mu.Lock()
		b.identity = ident
		b.pairing = false
		avail := b.available
		b.mu.Unlock()
		b.w.event("connected", map[string]any{"jid": ident.JID, "lid": ident.LID, "device": ident.Device})
		b.log.Printf("connected device=%d", ident.Device)
		if !b.cfg.DryRun {
			b.goSafe(func() {
				ctx, cancel := context.WithTimeout(context.Background(), netTimeout)
				defer cancel()
				if err := b.client.SetPresence(ctx, avail); err != nil {
					b.log.Printf("presence after connect failed: %v", err)
				}
			})
		}
	case e.Offline != "":
		b.mu.Lock()
		closing := b.closing
		b.syncing = false
		b.mu.Unlock()
		if !closing {
			b.w.event("offline", map[string]any{"reason": e.Offline})
			b.log.Printf("offline reason=%s", e.Offline)
		}
	case e.LoggedOut != "":
		b.mu.Lock()
		b.identity = Identity{}
		b.mu.Unlock()
		b.w.event("logged_out", map[string]any{"reason": e.LoggedOut})
		b.log.Printf("logged_out reason=%s", e.LoggedOut)
	case e.SyncStart:
		b.mu.Lock()
		b.syncing = true
		b.mu.Unlock()
	case e.SyncDone:
		b.mu.Lock()
		b.syncing = false
		b.mu.Unlock()
	case e.StoreError != "":
		b.w.event("error", map[string]any{"code": "store_error", "error": "the WhatsApp store failed"})
		b.log.Printf("store_error: %s", e.StoreError)
	case e.ChatChanged != "":
		b.chatChanged(e.ChatChanged)
	case e.Message != nil:
		b.inbound(e.Message)
	case e.HistoryNotice != nil:
		b.historyNotice(e.HistoryNotice)
	case e.Revoked != nil:
		b.revoked(e.Revoked)
	}
}

// revoked tells chatd that a message of an allowed chat was deleted, so it
// leaves the context Mac sends to the model. Never for a denied chat.
func (b *Bridge) revoked(r *Revoked) {
	b.mu.Lock()
	ok := b.allow[r.Chat] && !b.deny[r.Chat]
	b.mu.Unlock()
	if !ok {
		return
	}
	b.w.event("revoked", map[string]any{"chat": r.Chat, "msg": r.MsgID})
	b.log.Printf("revoked msg=%s", r.MsgID)
}

func (b *Bridge) chatChanged(chat string) {
	b.mu.Lock()
	delete(b.members, chat) // membership may have changed: re-read before the next mention
	ok := b.allow[chat] && !b.deny[chat]
	b.mu.Unlock()
	if !ok {
		return
	}
	b.goSafe(func() {
		ctx, cancel := context.WithTimeout(context.Background(), netTimeout)
		defer cancel()
		info, err := b.client.GetChat(ctx, chat)
		if err != nil {
			b.log.Printf("chat_changed lookup failed: %v", err)
			return
		}
		b.w.event("chat_changed", map[string]any{"chat": chat, "subject": info.Subject, "members": info.Members, "owner_only": info.OwnerOnly})
		b.log.Printf("chat_changed chat=%s members=%d owner_only=%v", chat, info.Members, info.OwnerOnly)
	})
}

// unsupportedKinds are never downloaded (protocol §4).
var unsupportedKinds = map[string]bool{"voice": true, "audio": true, "video": true, "sticker": true}

// inbound runs the §5 filter. A trigger candidate becomes a `message` event
// (images and documents downloaded); a non-owner message in a context chat
// becomes a `context` event (nothing downloaded); the rest is dropped.
func (b *Bridge) inbound(in *Inbound) {
	b.mu.Lock()
	if b.allow[in.Chat] && !b.deny[in.Chat] && in.MsgID != "" && in.SenderUser != "" {
		b.senders.put(MsgMeta{Chat: in.Chat, ID: in.MsgID, FromMe: in.IsFromMe, At: in.Timestamp}, in.SenderUser)
	}
	st := b.filterStateLocked()
	reason, asContext := filter(in, st)
	if reason != "" && !asContext {
		b.dropped[reason]++
	}
	replayed := in.Replayed || b.syncing
	b.mu.Unlock()
	if asContext {
		b.emitContext(in, st.self, replayed, false)
		return
	}
	if reason != "" {
		if reason == dropDenied || reason == dropNotAllowed {
			b.log.Printf("dropped{%s}", reason)
		} else {
			b.log.Printf("dropped{%s} msg=%s", reason, in.MsgID)
		}
		return
	}

	ev := map[string]any{
		"chat":          in.Chat,
		"msg":           in.MsgID,
		"sender_device": in.SenderDevice,
		"from_me":       in.IsFromMe, // raw, so chatd can re-check owner-ness (N10)
		"text":          in.Text,
		"sent_at":       float64(in.Timestamp.UnixMilli()) / 1000,
		"replayed":      replayed,
		"attachments":   []map[string]any{},
		"unsupported":   "",
	}
	if in.Quoted != nil {
		ev["quoted"] = b.quoted(in, st.self)
	}
	if in.Reaction != nil {
		ev["reaction"] = in.Reaction
		ev["text"] = ""
	}
	kind := ""
	if m := in.Media; m != nil {
		kind = m.Kind
		if unsupportedKinds[m.Kind] {
			ev["unsupported"] = m.Kind
		} else if m.Kind == "image" || m.Kind == "document" {
			ev["attachments"] = []map[string]any{b.download(in)}
		}
	}
	b.w.event("message", ev)
	b.log.Printf("message msg=%s device=%d len=%d media=%s replayed=%v", in.MsgID, in.SenderDevice, len(in.Text), kind, replayed)
}

func (b *Bridge) filterStateLocked() filterState {
	return filterState{
		allow: b.allow, context: b.context, deny: b.deny, trustAll: b.cfg.TrustedAll, trusted: b.trusted,
		self: b.identity, sent: b.sent.has, now: b.now(),
	}
}

// emitContext sends a `context` event: data for chatd's ring, never a
// trigger. Media is described by kind (and a document's name), never fetched.
func (b *Bridge) emitContext(in *Inbound, self Identity, replayed, history bool) {
	text := in.Text
	if r := []rune(text); len(r) > maxContextText {
		text = string(r[:maxContextText])
	}
	ev := map[string]any{
		"chat":        in.Chat,
		"msg":         in.MsgID,
		"sender":      in.SenderUser,
		"sender_name": in.SenderName,
		"from_owner":  in.IsFromMe && isSelf(in.SenderUser, self),
		"at":          float64(in.Timestamp.UnixMilli()) / 1000,
		"text":        text,
		"replayed":    replayed,
		"history":     history,
	}
	if m := in.Media; m != nil {
		media := map[string]any{"kind": m.Kind}
		if m.Kind == "document" && m.FileName != "" {
			media["name"] = m.FileName
		}
		ev["media"] = media
	}
	if in.Quoted != nil {
		ev["quoted_msg"] = in.Quoted.MsgID
	}
	b.w.event("context", ev)
	b.log.Printf("context msg=%s len=%d history=%v", in.MsgID, len(in.Text), history)
}

// quoted builds the extended `quoted` of a trigger (protocol §4). Only a quote
// of the same, non-denied chat carries content; anything else is other_chat.
func (b *Bridge) quoted(in *Inbound, self Identity) map[string]any {
	q := in.Quoted
	b.mu.Lock()
	foreign := (q.RemoteJID != "" && q.RemoteJID != in.Chat) || b.deny[in.Chat] || b.deny[q.RemoteJID]
	byUs := b.sent.has(q.MsgID)
	meta, known := b.senders.meta(in.Chat, q.MsgID)
	b.mu.Unlock()
	if foreign {
		b.log.Printf("quote from another chat msg=%s", in.MsgID)
		return map[string]any{"msg": q.MsgID, "kind": "other_chat"}
	}
	out := map[string]any{
		"msg":         q.MsgID,
		"sender":      q.Participant,
		"sender_name": q.SenderName,
		"from_owner":  isSelf(q.Participant, self) || (q.Participant == "" && byUs),
		"from_mac":    byUs,
		"at":          0.0,
		"text":        q.Text,
		"links":       extractLinks(q.Text, q.Links),
	}
	if known && !meta.At.IsZero() {
		out["at"] = float64(meta.At.UnixMilli()) / 1000
	}
	if m := q.Media; m != nil {
		media := map[string]any{"kind": m.Kind}
		if m.Kind == "image" || m.Kind == "document" {
			media["name"], media["mime"], media["size"] = m.FileName, m.Mime, m.Size
			if path, size := b.downloadQuoted(in.MsgID, m); path != "" {
				media["path"], media["size"] = path, size
			}
		}
		out["media"] = media
	}
	return out
}

var linkRe = regexp.MustCompile(`https?://[^\s<>"'\x60]+`)

// extractLinks returns the http(s) URLs of text plus extra (a link preview's
// URLs), de-duplicated, at most maxLinks.
func extractLinks(text string, extra []string) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, u := range append(linkRe.FindAllString(text, -1), extra...) {
		u = strings.TrimRight(u, ".,;:!?)]}")
		low := strings.ToLower(u)
		if (!strings.HasPrefix(low, "http://") && !strings.HasPrefix(low, "https://")) || seen[u] || len(u) > 2048 {
			continue
		}
		seen[u] = true
		out = append(out, u)
		if len(out) == maxLinks {
			break
		}
	}
	return out
}

// downloadQuoted fetches a quoted image or document because the owner's
// trigger quoted it. The cap applies to the bytes actually written; over it,
// the partial file is deleted and no path is returned. The result is checked
// with realpath to lie inside the media dir.
func (b *Bridge) downloadQuoted(msgID string, m *MediaIn) (string, int64) {
	if m.Handle == nil || m.Size > maxQuotedMedia {
		b.log.Printf("quoted media skipped msg=%s size=%d", msgID, m.Size)
		return "", 0
	}
	name := unsafeName.ReplaceAllString(m.FileName, "_")
	if name == "" || name == "." || name == ".." {
		name = m.Kind
	}
	if len(name) > 100 {
		name = name[len(name)-100:]
	}
	path := filepath.Join(b.cfg.MediaDir, "q-"+unsafeName.ReplaceAllString(msgID, "_")+"-"+name)
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		b.log.Printf("quoted media open failed msg=%s", msgID)
		return "", 0
	}
	// The encrypted body is the file plus padding and a 10-byte MAC.
	cf := &CappedFile{File: f, Cap: maxQuotedMedia + 64}
	ctx, cancel := context.WithTimeout(context.Background(), downloadTimeout)
	defer cancel()
	err = b.client.DownloadToFile(ctx, m, cf)
	var size int64
	if st, serr := f.Stat(); serr == nil {
		size = st.Size()
	}
	f.Close()
	if err != nil || size > maxQuotedMedia {
		_ = os.Remove(path)
		b.log.Printf("quoted media dropped msg=%s size=%d too_large=%v", msgID, size, errors.Is(err, ErrTooLarge) || size > maxQuotedMedia)
		return "", 0
	}
	real, err1 := filepath.EvalSymlinks(path)
	dir, err2 := filepath.EvalSymlinks(b.cfg.MediaDir)
	if err1 != nil || err2 != nil || filepath.Dir(real) != dir {
		_ = os.Remove(path)
		b.log.Printf("quoted media outside the media dir msg=%s", msgID)
		return "", 0
	}
	b.log.Printf("quoted media saved msg=%s kind=%s size=%d", msgID, m.Kind, size)
	return real, size
}

func (b *Bridge) sentHas(id string) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.sent.has(id)
}

var unsafeName = regexp.MustCompile(`[^A-Za-z0-9._-]+`)

// download fetches an image or document into --media-dir (0600).
func (b *Bridge) download(in *Inbound) map[string]any {
	m := in.Media
	att := map[string]any{"kind": m.Kind, "path": "", "mime": m.Mime, "name": m.FileName, "size": m.Size}
	if m.Size > maxMedia {
		att["error"] = "too_large"
		b.log.Printf("media skipped msg=%s size=%d (over cap)", in.MsgID, m.Size)
		return att
	}
	ctx, cancel := context.WithTimeout(context.Background(), downloadTimeout)
	defer cancel()
	data, err := b.client.Download(ctx, m)
	if err != nil {
		att["error"] = "download_failed"
		b.log.Printf("media download failed msg=%s: %v", in.MsgID, err)
		return att
	}
	name := unsafeName.ReplaceAllString(m.FileName, "_")
	if name == "" || name == "." || name == ".." {
		name = m.Kind
		if exts, _ := mime.ExtensionsByType(m.Mime); len(exts) > 0 {
			name += exts[0]
		}
	}
	idPart := unsafeName.ReplaceAllString(in.MsgID, "_")
	if len(name) > 100 {
		name = name[len(name)-100:]
	}
	path := filepath.Join(b.cfg.MediaDir, idPart+"-"+name)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		att["error"] = "write_failed"
		b.log.Printf("media write failed msg=%s", in.MsgID)
		return att
	}
	_ = os.Chmod(path, 0o600)
	att["path"] = path
	att["size"] = int64(len(data))
	b.log.Printf("media saved msg=%s kind=%s size=%d", in.MsgID, m.Kind, len(data))
	return att
}

// ---- members, mentions and history (protocol 2) -----------------------------

type membersEntry struct {
	list []Member
	at   time.Time
}

// chatMembers returns the chat's participants from the cache, or (fetch)
// from WhatsApp. The cache is dropped on every membership change.
func (b *Bridge) chatMembers(ctx context.Context, chat string, fetch bool) ([]Member, error) {
	b.mu.Lock()
	e, ok := b.members[chat]
	fresh := ok && b.now().Sub(e.at) < membersTTL
	b.mu.Unlock()
	if fresh {
		return e.list, nil
	}
	if !fetch {
		return nil, errors.New("members not cached")
	}
	list, err := b.client.Members(ctx, chat)
	if err != nil {
		return nil, err
	}
	if list == nil {
		list = []Member{}
	}
	b.mu.Lock()
	b.members[chat] = membersEntry{list: list, at: b.now()}
	b.mu.Unlock()
	return list, nil
}

// checkMentions keeps the JIDs that are current participants of chat and whose
// @<number> is in the text, at most maxMentions (protocol §3.1).
func (b *Bridge) checkMentions(ctx context.Context, chat, text string, jids []string, fetch bool) []string {
	kept := []string{}
	if len(jids) == 0 {
		return kept
	}
	list, err := b.chatMembers(ctx, chat, fetch)
	if err != nil {
		b.log.Printf("mentions dropped: no member list (%d asked)", len(jids))
		return kept
	}
	byJID := make(map[string]Member, len(list))
	for _, m := range list {
		byJID[m.JID] = m
	}
	seen := map[string]bool{}
	for _, j := range jids {
		m, ok := byJID[j]
		if !ok || seen[j] || !mentionIn(text, m.Number) {
			continue
		}
		seen[j] = true
		kept = append(kept, j)
		if len(kept) == maxMentions {
			break
		}
	}
	return kept
}

// mentionIn reports whether text holds "@<number>" not followed by a digit.
func mentionIn(text, number string) bool {
	if number == "" {
		return false
	}
	tok := "@" + number
	for i := 0; ; {
		k := strings.Index(text[i:], tok)
		if k < 0 {
			return false
		}
		end := i + k + len(tok)
		if end == len(text) || text[end] < '0' || text[end] > '9' {
			return true
		}
		i = end
	}
}

func (b *Bridge) membersOp(id int64, r *request) {
	if r.Chat == "" {
		b.fail(id, "bad_args", "chat is required")
		return
	}
	b.mu.Lock()
	var code, msg string
	switch {
	case b.deny[r.Chat]:
		code, msg = "denied", "that chat is on the never-list"
	case !b.allow[r.Chat]:
		code, msg = "not_allowed", "that chat is not allowlisted"
	default:
		code, msg = b.linkLocked()
	}
	b.mu.Unlock()
	if code != "" {
		b.fail(id, code, msg)
		return
	}
	chat := r.Chat
	b.async(func(ctx context.Context) {
		list, err := b.chatMembers(ctx, chat, true)
		if err != nil {
			b.netErr(id, "members", err)
			return
		}
		b.log.Printf("members id=%d n=%d", id, len(list)) // names never logged
		b.ok(id, map[string]any{"members": list})
	})
}

// history asks the owner's phone for up to count messages before before_msg
// in a context chat (W10.3). Every result is context only.
func (b *Bridge) history(id int64, r *request) {
	if r.Chat == "" || r.Before == "" || r.Count < 1 || r.Count > maxHistory {
		b.fail(id, "bad_args", "chat, before_msg and count (1-50) are required")
		return
	}
	if b.cfg.DryRun {
		b.fail(id, "bad_args", "history is off under --dry-run")
		return
	}
	var before MsgMeta
	extra := func() (string, string) { // runs under b.mu
		if !b.context[r.Chat] {
			return "not_allowed", "history is only for context chats"
		}
		m, ok := b.senders.meta(r.Chat, r.Before)
		if !ok {
			return "not_found", "unknown before_msg"
		}
		if last, ok := b.histLast[r.Chat]; ok && b.now().Sub(last) < historyEvery {
			return "rate", "history was requested for this chat within the hour"
		}
		before = m
		return "", ""
	}
	if !b.gate(id, r.Chat, bucketNone, extra) {
		return
	}
	chat, count := r.Chat, r.Count
	b.mu.Lock()
	now := b.now()
	b.histLast[chat], b.histOpen[chat] = now, now.Add(historyWait)
	b.mu.Unlock()
	b.async(func(ctx context.Context) {
		if err := b.client.RequestHistory(ctx, before, count); err != nil {
			b.mu.Lock()
			delete(b.histOpen, chat)
			b.mu.Unlock()
			b.netErr(id, "history", err)
			return
		}
		b.log.Printf("history id=%d count=%d", id, count)
		b.ok(id, map[string]any{"requested": true})
	})
}

// historyNotice handles an on-demand history blob: processed only while a
// request is open, only for that chat while it is still a context chat, and
// every message becomes context with history=true, whatever its sender.
func (b *Bridge) historyNotice(notice any) {
	b.mu.Lock()
	now := b.now()
	open := 0
	for c, deadline := range b.histOpen {
		if now.After(deadline) {
			delete(b.histOpen, c)
		} else {
			open++
		}
	}
	if open == 0 {
		b.dropped["history_unrequested"]++
	}
	b.mu.Unlock()
	if open == 0 {
		b.log.Printf("dropped{history_unrequested}")
		return
	}
	b.inflight.Add(1)
	b.goSafe(func() {
		defer b.inflight.Done()
		ctx, cancel := context.WithTimeout(context.Background(), downloadTimeout)
		defer cancel()
		msgs, err := b.client.FetchHistory(ctx, notice)
		if err != nil {
			b.log.Printf("history fetch failed: %v", err)
			return
		}
		emitted, done := 0, map[string]bool{}
		for _, in := range msgs {
			b.mu.Lock()
			deadline, ok := b.histOpen[in.Chat]
			ok = ok && !b.now().After(deadline) && b.context[in.Chat]
			st := b.filterStateLocked()
			reason, _ := filter(in, st)
			b.mu.Unlock()
			if !ok {
				b.countDrop("history_other_chat")
				continue
			}
			// Rules 1, 4 and 5 still apply; owner-ness does not make a trigger.
			if (reason != "" && reason != dropNotOwner && reason != dropUntrusted) || in.Reaction != nil {
				b.countDrop("history_" + reason)
				continue
			}
			in.Media = mediaKindOnly(in.Media)
			b.emitContext(in, st.self, false, true)
			emitted++
			done[in.Chat] = true
		}
		b.mu.Lock()
		for c := range done {
			delete(b.histOpen, c)
		}
		b.mu.Unlock()
		b.log.Printf("history processed n=%d emitted=%d", len(msgs), emitted)
	})
}

func (b *Bridge) countDrop(reason string) {
	b.mu.Lock()
	b.dropped[reason]++
	b.mu.Unlock()
}

// mediaKindOnly keeps what a context event may say about media.
func mediaKindOnly(m *MediaIn) *MediaIn {
	if m == nil {
		return nil
	}
	return &MediaIn{Kind: m.Kind, FileName: m.FileName}
}
