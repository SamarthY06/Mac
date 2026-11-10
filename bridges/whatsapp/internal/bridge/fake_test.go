package bridge

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeClient is an in-memory WhatsApp. It never touches the network.
type fakeClient struct {
	mu        sync.Mutex
	linked    bool
	connected bool
	ident     Identity
	handler   func(Event)
	nextID    int
	calls     []string // op names of every network write
	downloads int
	media     map[string][]byte // handle -> bytes
	pairItems []QRItem
	chats     []ChatInfo
	members   map[string][]Member // chat -> participants
	history   []*Inbound          // what FetchHistory returns
	histReqs  []MsgMeta
	sendMent  [][]string // mentions of every SendText/Edit
	written   int64      // bytes the last DownloadToFile managed to write
}

func newFake() *fakeClient {
	return &fakeClient{
		linked: true, connected: true,
		ident:   Identity{JID: "111@s.whatsapp.net", LID: "999@lid", Device: 7},
		media:   map[string][]byte{},
		members: map[string][]Member{},
	}
}

func (f *fakeClient) record(op string) {
	f.mu.Lock()
	f.calls = append(f.calls, op)
	f.mu.Unlock()
}

func (f *fakeClient) Calls() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string{}, f.calls...)
}

func (f *fakeClient) Linked() bool    { f.mu.Lock(); defer f.mu.Unlock(); return f.linked }
func (f *fakeClient) Connected() bool { f.mu.Lock(); defer f.mu.Unlock(); return f.connected }
func (f *fakeClient) Identity() Identity {
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.linked {
		return Identity{}
	}
	return f.ident
}
func (f *fakeClient) Connect() error {
	f.record("connect")
	go f.fire(Event{Connected: true})
	return nil
}
func (f *fakeClient) Disconnect() {}
func (f *fakeClient) Close()      {}
func (f *fakeClient) SetHandler(h func(Event)) {
	f.mu.Lock()
	f.handler = h
	f.mu.Unlock()
}
func (f *fakeClient) fire(e Event) {
	f.mu.Lock()
	h := f.handler
	f.mu.Unlock()
	h(e)
}
func (f *fakeClient) StartPair(ctx context.Context) (<-chan QRItem, error) {
	f.record("pair")
	ch := make(chan QRItem, len(f.pairItems))
	for _, it := range f.pairItems {
		ch <- it
	}
	close(ch)
	return ch, nil
}
func (f *fakeClient) Logout(ctx context.Context) error {
	f.record("logout")
	f.mu.Lock()
	f.linked, f.connected = false, false
	f.mu.Unlock()
	return nil
}
func (f *fakeClient) ListChats(ctx context.Context) ([]ChatInfo, error) {
	return f.chats, nil
}
func (f *fakeClient) GetChat(ctx context.Context, chat string) (ChatInfo, error) {
	for _, c := range f.chats {
		if c.ID == chat {
			return c, nil
		}
	}
	return ChatInfo{}, errors.New("no chat")
}
func (f *fakeClient) newID() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.nextID++
	return fmt.Sprintf("3EB0%04d", f.nextID)
}
func (f *fakeClient) SendText(ctx context.Context, chat, text string, reply *ReplyRef, mentions []string) (string, error) {
	op := "send_text"
	if reply != nil {
		op += "/reply:" + reply.MsgID + ":" + reply.Sender
	}
	f.record(op)
	f.mu.Lock()
	f.sendMent = append(f.sendMent, mentions)
	f.mu.Unlock()
	return f.newID(), nil
}
func (f *fakeClient) Edit(ctx context.Context, chat, msgID, text string, mentions []string) error {
	f.record("edit")
	f.mu.Lock()
	f.sendMent = append(f.sendMent, mentions)
	f.mu.Unlock()
	return nil
}
func (f *fakeClient) Members(ctx context.Context, chat string) ([]Member, error) {
	f.record("members")
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.members[chat], nil
}
func (f *fakeClient) RequestHistory(ctx context.Context, before MsgMeta, count int) error {
	f.record(fmt.Sprintf("history:%d", count))
	f.mu.Lock()
	f.histReqs = append(f.histReqs, before)
	f.mu.Unlock()
	return nil
}
func (f *fakeClient) FetchHistory(ctx context.Context, notice any) ([]*Inbound, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.history, nil
}

// DownloadToFile writes the handle's bytes in 1 MiB chunks, like a stream,
// so a body larger than declared is cut off by the CappedFile.
func (f *fakeClient) DownloadToFile(ctx context.Context, m *MediaIn, w *CappedFile) error {
	f.mu.Lock()
	f.downloads++
	key, _ := m.Handle.(string)
	data := f.media[key]
	f.mu.Unlock()
	var n int64
	defer func() { f.mu.Lock(); f.written = n; f.mu.Unlock() }()
	for len(data) > 0 {
		k := min(len(data), 1<<20)
		if _, err := w.Write(data[:k]); err != nil {
			return err
		}
		n += int64(k)
		data = data[k:]
	}
	return nil
}
func (f *fakeClient) React(ctx context.Context, chat, sender, msgID, emoji string) error {
	f.record("react:" + sender)
	return nil
}
func (f *fakeClient) SendMedia(ctx context.Context, m MediaOut) (string, error) {
	f.record("send_media:" + m.Kind)
	return f.newID(), nil
}
func (f *fakeClient) SetPresence(ctx context.Context, available bool) error {
	f.record(fmt.Sprintf("presence:%v", available))
	return nil
}
func (f *fakeClient) ChatPresence(ctx context.Context, chat string, composing bool) error {
	f.record("chat_presence")
	return nil
}
func (f *fakeClient) Download(ctx context.Context, m *MediaIn) ([]byte, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.downloads++
	key, _ := m.Handle.(string)
	return f.media[key], nil
}

// ---- harness ----------------------------------------------------------------

type clock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *clock) Now() time.Time { c.mu.Lock(); defer c.mu.Unlock(); return c.t }
func (c *clock) Advance(d time.Duration) {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
}

type harness struct {
	t      *testing.T
	fake   *fakeClient
	clock  *clock
	in     *io.PipeWriter
	frames chan map[string]any
	stderr *syncBuf
	code   chan int
	opened int
	store  string
	media  string
	mu     sync.Mutex
	seen   []map[string]any
}

type syncBuf struct {
	mu sync.Mutex
	b  strings.Builder
}

func (s *syncBuf) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}
func (s *syncBuf) String() string { s.mu.Lock(); defer s.mu.Unlock(); return s.b.String() }

// start runs the bridge in-process with extra flags. --store and --media-dir
// point into a temp dir unless given.
func start(t *testing.T, fake *fakeClient, flags ...string) *harness {
	t.Helper()
	dir := t.TempDir()
	h := &harness{
		t: t, fake: fake,
		clock:  &clock{t: time.Date(2026, 11, 10, 12, 0, 0, 0, time.UTC)},
		frames: make(chan map[string]any, 256),
		stderr: &syncBuf{},
		code:   make(chan int, 1),
		store:  filepath.Join(dir, "wa", "device.db"),
		media:  filepath.Join(dir, "media"),
	}
	args := append([]string{"--store", h.store, "--media-dir", h.media}, flags...)
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	h.in = inW
	go func() {
		sc := bufio.NewScanner(outR)
		sc.Buffer(make([]byte, 1<<20), 4<<20)
		for sc.Scan() {
			var m map[string]any
			if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
				t.Errorf("bridge wrote non-JSON: %q", sc.Text())
				continue
			}
			h.frames <- m
		}
		close(h.frames)
	}()
	open := func(cfg Config, log *Logger) (Client, error) {
		h.opened++
		return fake, nil
	}
	go func() {
		c := Run(args, inR, outW, h.stderr, Options{Open: open, Now: h.clock.Now})
		outW.Close()
		inR.Close()
		h.code <- c
	}()
	t.Cleanup(func() { inW.Close() })
	return h
}

func (h *harness) send(v map[string]any) {
	h.t.Helper()
	b, _ := json.Marshal(v)
	h.sendRaw(string(b) + "\n")
}

func (h *harness) sendRaw(s string) {
	h.t.Helper()
	if _, err := io.WriteString(h.in, s); err != nil {
		h.t.Fatalf("write stdin: %v", err)
	}
}

// next returns the next frame matching pred (frames that do not match are
// kept for later lookups).
func (h *harness) next(desc string, pred func(map[string]any) bool) map[string]any {
	h.t.Helper()
	h.mu.Lock()
	for i, m := range h.seen {
		if pred(m) {
			h.seen = append(h.seen[:i], h.seen[i+1:]...)
			h.mu.Unlock()
			return m
		}
	}
	h.mu.Unlock()
	timeout := time.After(3 * time.Second)
	for {
		select {
		case m, ok := <-h.frames:
			if !ok {
				h.t.Fatalf("stdout closed while waiting for %s", desc)
			}
			if pred(m) {
				return m
			}
			h.mu.Lock()
			h.seen = append(h.seen, m)
			h.mu.Unlock()
		case <-timeout:
			h.t.Fatalf("timed out waiting for %s", desc)
		}
	}
}

func (h *harness) resp(id int) map[string]any {
	h.t.Helper()
	return h.next(fmt.Sprintf("response %d", id), func(m map[string]any) bool {
		v, ok := m["id"].(float64)
		return ok && int(v) == id
	})
}

func (h *harness) nullResp() map[string]any {
	h.t.Helper()
	return h.next("id:null response", func(m map[string]any) bool {
		v, has := m["id"]
		return has && v == nil
	})
}

func (h *harness) event(name string) map[string]any {
	h.t.Helper()
	return h.next("event "+name, func(m map[string]any) bool { return m["ev"] == name })
}

// noEvent asserts no event called name arrives within d.
func (h *harness) noEvent(name string, d time.Duration) {
	h.t.Helper()
	h.mu.Lock()
	for _, m := range h.seen {
		if m["ev"] == name {
			h.mu.Unlock()
			h.t.Fatalf("unexpected event %s: %v", name, m)
		}
	}
	h.mu.Unlock()
	deadline := time.After(d)
	for {
		select {
		case m, ok := <-h.frames:
			if !ok {
				return
			}
			if m["ev"] == name {
				h.t.Fatalf("unexpected event %s: %v", name, m)
			}
			h.mu.Lock()
			h.seen = append(h.seen, m)
			h.mu.Unlock()
		case <-deadline:
			return
		}
	}
}

func (h *harness) call(id int, op string, args map[string]any) map[string]any {
	h.t.Helper()
	m := map[string]any{"id": id, "op": op}
	for k, v := range args {
		m[k] = v
	}
	h.send(m)
	return h.resp(id)
}

func (h *harness) hello() map[string]any {
	h.t.Helper()
	h.event("ready")
	r := h.call(0, "hello", map[string]any{"protocol": "2"})
	if r["ok"] != true {
		h.t.Fatalf("hello failed: %v", r)
	}
	return r["data"].(map[string]any)
}

func (h *harness) exitCode(within time.Duration) int {
	h.t.Helper()
	select {
	case c := <-h.code:
		return c
	case <-time.After(within):
		h.t.Fatalf("bridge did not exit within %v", within)
		return -1
	}
}

func wantOK(t *testing.T, r map[string]any) map[string]any {
	t.Helper()
	if r["ok"] != true {
		t.Fatalf("want ok, got %v", r)
	}
	d, _ := r["data"].(map[string]any)
	return d
}

func wantErr(t *testing.T, r map[string]any, code string) {
	t.Helper()
	if r["ok"] != false || r["code"] != code {
		t.Fatalf("want error %s, got %v", code, r)
	}
}
