package bridge

import (
	"bytes"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var allowFlags = []string{"--allow", group, "--deny", never}

// ---- flags ------------------------------------------------------------------

func TestFlagValidation(t *testing.T) {
	base := []string{"--store", "/tmp/x/device.db", "--media-dir", "/tmp/x/media"}
	bad := map[string][]string{
		"test without budget":   {"--test"},
		"allow overlaps deny":   {"--allow", group, "--deny", group},
		"bad jid":               {"--allow", "not-a-jid"},
		"jid with device":       {"--allow", "111:3@s.whatsapp.net"},
		"negative budget":       {"--budget", "-1"},
		"non-numeric budget":    {"--budget", "lots"},
		"zero max-per-min":      {"--max-per-min", "0"},
		"bad trusted device":    {"--trusted-devices", "x"},
		"context not in allow":  {"--allow", group, "--context", stranger},
		"context without allow": {"--context", group},
		"unknown flag":          {"--raise-budget"},
		"stray argument":        {"extra"},
	}
	for name, extra := range bad {
		if _, err := ParseFlags(append(append([]string{}, base...), extra...), io.Discard); err == nil {
			t.Errorf("%s: want error", name)
		}
	}
	if _, err := ParseFlags([]string{"--media-dir", "/tmp/m"}, io.Discard); err == nil {
		t.Error("missing --store: want error")
	}
	c, err := ParseFlags(append(base, "--test", "--budget", "4", "--allow", group+","+group, "--deny", never), io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if c.Budget != 4 || !c.Test || len(c.Allow) != 1 || c.MaxPerMin != 20 || !c.TrustedAll || len(c.TrustedDevices) != 0 {
		t.Fatalf("unexpected config %+v", c)
	}
	c, err = ParseFlags(append(base, "--allow", group, "--context", group, "--trusted-devices", "0,3"), io.Discard)
	if err != nil || c.TrustedAll || len(c.TrustedDevices) != 2 || len(c.Context) != 1 {
		t.Fatalf("context and device list: %+v %v", c, err)
	}
	c, _ = ParseFlags(base, io.Discard)
	if c.HasBudget() || len(c.Allow) != 0 {
		t.Fatalf("defaults: %+v", c)
	}
}

func TestBadFlagsExit2(t *testing.T) {
	h := start(t, newFake(), "--test") // no --budget
	if c := h.exitCode(2 * time.Second); c != ExitBadArgs {
		t.Fatalf("exit %d, want 2", c)
	}
	if h.opened != 0 {
		t.Fatal("store opened despite bad flags")
	}
}

// ---- hello, framing ---------------------------------------------------------

func TestHelloAndReady(t *testing.T) {
	h := start(t, newFake(), append(allowFlags, "--budget", "5")...)
	ready := h.event("ready")
	if ready["protocol"] != "2" || ready["linked"] != true {
		t.Fatalf("ready = %v", ready)
	}
	d := wantOK(t, h.call(1, "hello", map[string]any{"protocol": "2"}))
	if d["jid"] != ownerPN || d["lid"] != ownerLID || d["device"] != float64(7) || d["budget_left"] != float64(5) ||
		d["dry_run"] != false || d["test"] != false {
		t.Fatalf("hello data = %v", d)
	}
	if a := d["allow"].([]any); len(a) != 1 || a[0] != group {
		t.Fatalf("allow = %v", a)
	}
	if dn := d["deny"].([]any); len(dn) != 1 || dn[0] != never {
		t.Fatalf("deny = %v", dn)
	}
}

func TestHelloMustComeFirst(t *testing.T) {
	h := start(t, newFake(), allowFlags...)
	h.event("ready")
	wantErr(t, h.call(1, "status", nil), "bad_args")
	wantErr(t, h.call(2, "frobnicate", nil), "unknown_op")
	wantOK(t, h.call(3, "hello", map[string]any{"protocol": "2"}))
	wantOK(t, h.call(4, "status", nil))
}

func TestVersionMismatchExits2(t *testing.T) {
	h := start(t, newFake(), allowFlags...)
	h.event("ready")
	wantErr(t, h.call(1, "hello", map[string]any{"protocol": "1"}), "version_mismatch")
	// Anything after the mismatch must not be handled (the pipe may already be closed).
	io.WriteString(h.in, `{"id":2,"op":"status"}`+"\n")
	if c := h.exitCode(3 * time.Second); c != ExitBadArgs {
		t.Fatalf("exit %d, want 2", c)
	}
	for m := range h.frames {
		if m["id"] == float64(2) {
			t.Fatalf("command handled after version_mismatch: %v", m)
		}
	}
}

func TestFramingErrors(t *testing.T) {
	h := start(t, newFake(), allowFlags...)
	h.hello()

	h.sendRaw("{not json\n")
	r := h.nullResp()
	wantErr(t, r, "bad_frame")

	h.sendRaw("[1,2,3]\n")
	wantErr(t, h.nullResp(), "bad_frame")

	h.sendRaw(`{"op":"status"}` + "\n")
	wantErr(t, h.nullResp(), "bad_frame")

	h.sendRaw(`{"id":"seven","op":"status"}` + "\n")
	wantErr(t, h.nullResp(), "bad_frame")

	// Over 1 MiB: rejected, and the stream resyncs at the newline.
	big := `{"id":5,"op":"send_text","chat":"` + group + `","text":"` + strings.Repeat("a", MaxLine) + `"}` + "\n"
	h.sendRaw(big)
	wantErr(t, h.nullResp(), "bad_frame")

	wantErr(t, h.call(6, "no_such_op", nil), "unknown_op")
	wantErr(t, h.call(7, "send_text", map[string]any{"chat": 42, "text": "x"}), "bad_args")
	wantErr(t, h.call(8, "send_text", map[string]any{"chat": group}), "bad_args")
	// Unknown fields are ignored.
	wantOK(t, h.call(9, "status", map[string]any{"whatever": true}))
	// Blank lines are skipped.
	h.sendRaw("\n\n")
	wantOK(t, h.call(10, "status", nil))
	if calls := h.fake.Calls(); containsPrefix(calls, "send_text") {
		t.Fatalf("a rejected frame reached the network: %v", calls)
	}
}

func containsPrefix(xs []string, p string) bool {
	for _, x := range xs {
		if strings.HasPrefix(x, p) {
			return true
		}
	}
	return false
}

func countPrefix(xs []string, p string) int {
	n := 0
	for _, x := range xs {
		if strings.HasPrefix(x, p) {
			n++
		}
	}
	return n
}

// ---- immutability, test mode -------------------------------------------------

func TestDenyAndBudgetImmutable(t *testing.T) {
	h := start(t, newFake(), append(allowFlags, "--budget", "2")...)
	h.hello()
	// set_allow with a denied JID: denied, and nothing changes.
	wantErr(t, h.call(1, "set_allow", map[string]any{"allow": []string{stranger, never}}), "denied")
	d := wantOK(t, h.call(2, "hello", map[string]any{"protocol": "2"}))
	if a := d["allow"].([]any); len(a) != 1 || a[0] != group {
		t.Fatalf("allow changed after a denied set_allow: %v", a)
	}
	// No command can alter --deny or --budget: extra fields are ignored and
	// there is no op for it.
	wantOK(t, h.call(3, "set_allow", map[string]any{"allow": []string{group}, "deny": []string{}, "budget": 1000}))
	for i, op := range []string{"set_deny", "set_budget", "reset_budget", "raise_budget"} {
		wantErr(t, h.call(10+i, op, map[string]any{"deny": []string{}, "budget": 1000}), "unknown_op")
	}
	wantOK(t, h.call(20, "status", map[string]any{"budget": 1000}))
	d = wantOK(t, h.call(21, "hello", map[string]any{"protocol": "2", "budget": 99, "deny": []string{}}))
	if dn := d["deny"].([]any); len(dn) != 1 || dn[0] != never || d["budget_left"] != float64(2) {
		t.Fatalf("deny/budget changed: %v", d)
	}
	// The never-listed chat stays unreachable.
	wantErr(t, h.call(22, "send_text", map[string]any{"chat": never, "text": "hi"}), "denied")
	if containsPrefix(h.fake.Calls(), "send_text") {
		t.Fatal("send to never-listed chat reached the network")
	}
}

func TestSetAllow(t *testing.T) {
	h := start(t, newFake(), "--deny", never)
	h.hello()
	wantErr(t, h.call(1, "send_text", map[string]any{"chat": group, "text": "hi"}), "not_allowed")
	d := wantOK(t, h.call(2, "set_allow", map[string]any{"allow": []string{group}}))
	if a := d["allow"].([]any); len(a) != 1 || a[0] != group {
		t.Fatalf("allow = %v", a)
	}
	wantOK(t, h.call(3, "send_text", map[string]any{"chat": group, "text": "hi"}))
	wantErr(t, h.call(4, "set_allow", map[string]any{"allow": []string{"bogus"}}), "bad_args")
	wantErr(t, h.call(5, "set_allow", nil), "bad_args")
}

func TestTestModeRefusals(t *testing.T) {
	f := newFake()
	h := start(t, f, append(allowFlags, "--test", "--budget", "4")...)
	h.hello()
	wantErr(t, h.call(1, "set_allow", map[string]any{"allow": []string{group, stranger}}), "test_mode")
	wantErr(t, h.call(2, "pair", nil), "test_mode")
	wantErr(t, h.call(3, "logout", nil), "test_mode")
	d := wantOK(t, h.call(4, "hello", map[string]any{"protocol": "2"}))
	if a := d["allow"].([]any); len(a) != 1 || d["test"] != true {
		t.Fatalf("hello = %v", d)
	}
	if containsPrefix(f.Calls(), "pair") || containsPrefix(f.Calls(), "logout") {
		t.Fatalf("refused op reached the client: %v", f.Calls())
	}
}

// ---- budget, rate --------------------------------------------------------------

func TestBudgetCountsAndIsFatal(t *testing.T) {
	f := newFake()
	h := start(t, f, append(allowFlags, "--budget", "4")...)
	h.hello()
	sent := wantOK(t, h.call(1, "send_text", map[string]any{"chat": group, "text": "Mac · one"}))
	msg := sent["msg"].(string)
	wantOK(t, h.call(2, "edit", map[string]any{"chat": group, "msg": msg, "text": "Mac · two"}))
	wantOK(t, h.call(3, "react", map[string]any{"chat": group, "msg": msg, "from_me": true, "emoji": "👀"}))
	// Not counted: presence, chat_presence, status, list_chats, refusals.
	wantOK(t, h.call(4, "presence", map[string]any{"state": "unavailable"}))
	wantOK(t, h.call(5, "chat_presence", map[string]any{"chat": group, "state": "composing"}))
	wantOK(t, h.call(6, "list_chats", nil))
	wantErr(t, h.call(7, "send_text", map[string]any{"chat": never, "text": "x"}), "denied")
	wantErr(t, h.call(8, "send_text", map[string]any{"chat": stranger, "text": "x"}), "not_allowed")
	wantErr(t, h.call(9, "edit", map[string]any{"chat": group, "msg": "nope", "text": "x"}), "not_found")
	st := wantOK(t, h.call(10, "status", nil))
	if st["budget_left"] != float64(1) || st["total_outbound"] != float64(3) {
		t.Fatalf("status = %v", st)
	}
	// Action 4 (send_media) is the last one allowed.
	file := filepath.Join(t.TempDir(), "cv.pdf")
	os.WriteFile(file, []byte("%PDF-1.4"), 0o600)
	wantOK(t, h.call(11, "send_media", map[string]any{"chat": group, "path": file, "kind": "document"}))
	// N+1: budget, budget_exhausted, exit 3; nothing after it runs.
	wantErr(t, h.call(12, "react", map[string]any{"chat": group, "msg": msg, "from_me": true, "emoji": "✅"}), "budget")
	ev := h.event("budget_exhausted")
	if ev["used"] != float64(4) {
		t.Fatalf("budget_exhausted = %v", ev)
	}
	io.WriteString(h.in, `{"id":13,"op":"send_text","chat":"`+group+`","text":"after"}`+"\n") // may hit a closed pipe
	if c := h.exitCode(3 * time.Second); c != ExitBudget {
		t.Fatalf("exit %d, want 3", c)
	}
	calls := f.Calls()
	if n := countPrefix(calls, "send_text") + countPrefix(calls, "edit") + countPrefix(calls, "react") + countPrefix(calls, "send_media"); n != 4 {
		t.Fatalf("network writes = %d (%v), want 4", n, calls)
	}
}

func TestBudgetZeroIsImmediatelyFatal(t *testing.T) {
	h := start(t, newFake(), append(allowFlags, "--test", "--budget", "0")...)
	h.hello()
	wantErr(t, h.call(1, "send_text", map[string]any{"chat": group, "text": "x"}), "budget")
	h.event("budget_exhausted")
	if c := h.exitCode(3 * time.Second); c != ExitBudget {
		t.Fatalf("exit %d, want 3", c)
	}
}

func TestCheckOrder(t *testing.T) {
	// deny before allow before rate before budget before link.
	f := newFake()
	f.connected = false
	h := start(t, f, append(allowFlags, "--budget", "1", "--max-per-min", "1")...)
	h.hello()
	wantErr(t, h.call(1, "send_text", map[string]any{"chat": never, "text": "x"}), "denied")
	wantErr(t, h.call(2, "send_text", map[string]any{"chat": stranger, "text": "x"}), "not_allowed")
	// Offline passes 1-4: it is counted only if it passes 1-5, so it is not.
	wantErr(t, h.call(3, "send_text", map[string]any{"chat": group, "text": "x"}), "offline")
	st := wantOK(t, h.call(4, "status", nil))
	if st["budget_left"] != float64(1) || st["sent_last_min"] != float64(0) {
		t.Fatalf("a refused command was counted: %v", st)
	}
	f.mu.Lock()
	f.connected = true
	f.mu.Unlock()
	wantOK(t, h.call(5, "send_text", map[string]any{"chat": group, "text": "x"}))
	// Rate (cap 1) is checked before budget (spent): rate, not fatal.
	wantErr(t, h.call(6, "send_text", map[string]any{"chat": group, "text": "x"}), "rate")
	h.noEvent("budget_exhausted", 100*time.Millisecond)
	h.clock.Advance(61 * time.Second)
	wantErr(t, h.call(7, "send_text", map[string]any{"chat": group, "text": "x"}), "budget")
	h.event("budget_exhausted")
	if c := h.exitCode(3 * time.Second); c != ExitBudget {
		t.Fatalf("exit %d, want 3", c)
	}
}

func TestNotLinked(t *testing.T) {
	f := newFake()
	f.linked, f.connected = false, false
	h := start(t, f, allowFlags...)
	d := h.hello()
	if d["linked"] != false || d["jid"] != nil {
		t.Fatalf("hello = %v", d)
	}
	wantErr(t, h.call(1, "send_text", map[string]any{"chat": group, "text": "x"}), "not_linked")
	wantErr(t, h.call(2, "list_chats", nil), "not_linked")
	wantErr(t, h.call(3, "logout", nil), "not_linked")
	if containsPrefix(f.Calls(), "connect") {
		t.Fatal("connected without a linked device")
	}
}

func TestRateLimits(t *testing.T) {
	f := newFake()
	h := start(t, f, append(allowFlags, "--max-per-min", "3")...)
	h.hello()
	var last string
	for i := 1; i <= 3; i++ {
		last = wantOK(t, h.call(i, "send_text", map[string]any{"chat": group, "text": "x"}))["msg"].(string)
	}
	wantErr(t, h.call(4, "send_text", map[string]any{"chat": group, "text": "x"}), "rate")
	file := filepath.Join(t.TempDir(), "a.png")
	os.WriteFile(file, []byte{0x89, 'P', 'N', 'G'}, 0o600)
	wantErr(t, h.call(5, "send_media", map[string]any{"chat": group, "path": file, "kind": "image"}), "rate")
	// Edits and reactions have their own cap (60/min), unaffected by sends.
	for i := 0; i < 60; i++ {
		wantOK(t, h.call(100+i, "react", map[string]any{"chat": group, "msg": last, "from_me": true, "emoji": "👀"}))
	}
	wantErr(t, h.call(200, "edit", map[string]any{"chat": group, "msg": last, "text": "y"}), "rate")
	st := wantOK(t, h.call(201, "status", nil))
	if st["sent_last_min"] != float64(3) {
		t.Fatalf("status = %v", st)
	}
	// Rolling window.
	h.clock.Advance(30 * time.Second)
	wantErr(t, h.call(202, "send_text", map[string]any{"chat": group, "text": "x"}), "rate")
	h.clock.Advance(31 * time.Second)
	wantOK(t, h.call(203, "send_text", map[string]any{"chat": group, "text": "x"}))
	wantOK(t, h.call(204, "edit", map[string]any{"chat": group, "msg": last, "text": "y"}))
	if st, _ := wantOK(t, h.call(205, "status", nil))["budget_left"]; st != nil {
		t.Fatal("budget_left present without --budget")
	}
}

// ---- outbound details ------------------------------------------------------------

func TestEditRules(t *testing.T) {
	h := start(t, newFake(), allowFlags...)
	h.hello()
	msg := wantOK(t, h.call(1, "send_text", map[string]any{"chat": group, "text": "x"}))["msg"].(string)
	wantErr(t, h.call(2, "edit", map[string]any{"chat": group, "msg": "someone-else", "text": "y"}), "not_found")
	h.clock.Advance(14 * time.Minute)
	wantOK(t, h.call(3, "edit", map[string]any{"chat": group, "msg": msg, "text": "y"}))
	h.clock.Advance(2 * time.Minute)
	wantErr(t, h.call(4, "edit", map[string]any{"chat": group, "msg": msg, "text": "z"}), "edit_window")
}

func TestReactAndReplyResolveParticipant(t *testing.T) {
	f := newFake()
	h := start(t, f, allowFlags...)
	h.hello()
	// A non-owner message in the allowed chat is dropped, but its author is
	// remembered (id and JID only) for reply/react with from_me=false.
	f.fire(Event{Message: &Inbound{Chat: group, MsgID: "OTHER1", SenderUser: "222@s.whatsapp.net", Timestamp: h.clock.Now(), Text: "hello"}})
	wantOK(t, h.call(1, "react", map[string]any{"chat": group, "msg": "OTHER1", "from_me": false, "emoji": "👍"}))
	wantErr(t, h.call(2, "react", map[string]any{"chat": group, "msg": "UNKNOWN", "from_me": false, "emoji": "👍"}), "not_found")
	wantOK(t, h.call(3, "react", map[string]any{"chat": group, "msg": "ANY", "from_me": true, "emoji": ""}))
	wantOK(t, h.call(4, "send_text", map[string]any{"chat": group, "text": "x", "reply_to": map[string]any{"msg": "OWN1", "from_me": true}}))
	wantOK(t, h.call(5, "send_text", map[string]any{"chat": group, "text": "x", "reply_to": map[string]any{"msg": "OTHER1", "from_me": false}}))
	calls := f.Calls()
	for _, want := range []string{"react:222@s.whatsapp.net", "react:", "send_text/reply:OWN1:", "send_text/reply:OTHER1:222@s.whatsapp.net"} {
		found := false
		for _, c := range calls {
			if c == want {
				found = true
			}
		}
		if !found {
			t.Fatalf("missing call %q in %v", want, calls)
		}
	}
}

func TestSendMediaChecks(t *testing.T) {
	f := newFake()
	h := start(t, f, allowFlags...)
	h.hello()
	dir := t.TempDir()
	wantErr(t, h.call(1, "send_media", map[string]any{"chat": group, "path": dir, "kind": "document"}), "bad_args")
	wantErr(t, h.call(2, "send_media", map[string]any{"chat": group, "path": filepath.Join(dir, "missing"), "kind": "document"}), "bad_args")
	wantErr(t, h.call(3, "send_media", map[string]any{"chat": group, "path": filepath.Join(dir, "x"), "kind": "video"}), "bad_args")
	big := filepath.Join(dir, "big.bin")
	fh, _ := os.Create(big)
	fh.Truncate(maxMedia + 1) // sparse
	fh.Close()
	wantErr(t, h.call(4, "send_media", map[string]any{"chat": group, "path": big, "kind": "document"}), "too_large")
	ok := filepath.Join(dir, "ok.png")
	os.WriteFile(ok, []byte("png"), 0o600)
	wantOK(t, h.call(5, "send_media", map[string]any{"chat": group, "path": ok, "kind": "image", "caption": "c"}))
	if countPrefix(f.Calls(), "send_media") != 1 {
		t.Fatalf("calls = %v", f.Calls())
	}
}

func TestDryRun(t *testing.T) {
	f := newFake()
	h := start(t, f, append(allowFlags, "--dry-run", "--budget", "10")...)
	h.hello()
	m1 := wantOK(t, h.call(1, "send_text", map[string]any{"chat": group, "text": "x"}))["msg"]
	m2 := wantOK(t, h.call(2, "send_text", map[string]any{"chat": group, "text": "y"}))["msg"]
	if m1 != "dry-1" || m2 != "dry-2" {
		t.Fatalf("ids %v %v", m1, m2)
	}
	wantOK(t, h.call(3, "edit", map[string]any{"chat": group, "msg": "dry-1", "text": "z"}))
	wantOK(t, h.call(4, "react", map[string]any{"chat": group, "msg": "dry-1", "from_me": true, "emoji": "✅"}))
	wantOK(t, h.call(5, "presence", map[string]any{"state": "available"}))
	wantOK(t, h.call(6, "chat_presence", map[string]any{"chat": group, "state": "composing"}))
	file := filepath.Join(t.TempDir(), "a.pdf")
	os.WriteFile(file, []byte("x"), 0o600)
	if m := wantOK(t, h.call(7, "send_media", map[string]any{"chat": group, "path": file, "kind": "document"}))["msg"]; m != "dry-3" {
		t.Fatalf("media id %v", m)
	}
	// Budget still counts under dry-run.
	if st := wantOK(t, h.call(8, "status", nil)); st["budget_left"] != float64(5) {
		t.Fatalf("status = %v", st)
	}
	for _, c := range f.Calls() {
		if c != "connect" {
			t.Fatalf("network write under --dry-run: %s", c)
		}
	}
}

func TestPresenceAfterConnect(t *testing.T) {
	f := newFake()
	h := start(t, f, allowFlags...)
	h.hello()
	h.event("connected")
	deadline := time.Now().Add(2 * time.Second)
	for !containsPrefix(f.Calls(), "presence:false") {
		if time.Now().After(deadline) {
			t.Fatalf("no unavailable presence after connect: %v", f.Calls())
		}
		time.Sleep(10 * time.Millisecond)
	}
	wantOK(t, h.call(1, "presence", map[string]any{"state": "available"}))
	wantErr(t, h.call(2, "presence", map[string]any{"state": "busy"}), "bad_args")
	wantErr(t, h.call(3, "chat_presence", map[string]any{"chat": never, "state": "composing"}), "denied")
	wantErr(t, h.call(4, "chat_presence", map[string]any{"chat": stranger, "state": "composing"}), "not_allowed")
}

// ---- pairing, logout, chats ---------------------------------------------------------

func TestPairFlow(t *testing.T) {
	f := newFake()
	f.linked, f.connected = false, false
	f.pairItems = []QRItem{{Code: "2@abc", Timeout: 60 * time.Second}, {Code: "2@def", Timeout: 20 * time.Second}, {Done: "timeout"}}
	h := start(t, f, allowFlags...)
	h.hello()
	wantOK(t, h.call(1, "pair", nil))
	q := h.event("qr")
	if q["code"] != "2@abc" || q["expires_s"] != float64(60) {
		t.Fatalf("qr = %v", q)
	}
	if q2 := h.event("qr"); q2["code"] != "2@def" {
		t.Fatalf("qr2 = %v", q2)
	}
	if e := h.event("error"); e["code"] != "pair_timeout" {
		t.Fatalf("error = %v", e)
	}
	// Linked: pair is refused.
	f.mu.Lock()
	f.linked, f.connected = true, true
	f.mu.Unlock()
	wantErr(t, h.call(2, "pair", nil), "already_linked")
	f.fire(Event{Connected: true})
	if c := h.event("connected"); c["jid"] != ownerPN || c["device"] != float64(7) {
		t.Fatalf("connected = %v", c)
	}
}

func TestLogoutStaysAlive(t *testing.T) {
	f := newFake()
	h := start(t, f, allowFlags...)
	h.hello()
	wantOK(t, h.call(1, "logout", nil))
	if e := h.event("logged_out"); e["reason"] != "logout" {
		t.Fatalf("logged_out = %v", e)
	}
	st := wantOK(t, h.call(2, "status", nil))
	if st["linked"] != false {
		t.Fatalf("status = %v", st)
	}
	wantErr(t, h.call(3, "send_text", map[string]any{"chat": group, "text": "x"}), "not_linked")
	wantOK(t, h.call(4, "pair", nil)) // ready to link again
	f.fire(Event{LoggedOut: "phone"})
	if e := h.event("logged_out"); e["reason"] != "phone" {
		t.Fatalf("logged_out = %v", e)
	}
	select {
	case c := <-h.code:
		t.Fatalf("bridge exited %d after logout", c)
	case <-time.After(100 * time.Millisecond):
	}
}

func TestListChatsAndChatChanged(t *testing.T) {
	f := newFake()
	f.chats = []ChatInfo{
		{ID: ownerPN, Subject: "Message yourself", Members: 1, OwnerOnly: true, IsSelf: true},
		{ID: group, Subject: "Interview Prep", Members: 1, OwnerOnly: true},
		{ID: never, Subject: "Timetable", Members: 30},
	}
	h := start(t, f, allowFlags...)
	h.hello()
	d := wantOK(t, h.call(1, "list_chats", nil))
	chats := d["chats"].([]any)
	if len(chats) != 3 {
		t.Fatalf("chats = %v", chats)
	}
	first := chats[0].(map[string]any)
	if first["is_self"] != true || first["owner_only"] != true || first["members"] != float64(1) {
		t.Fatalf("self chat = %v", first)
	}
	f.chats[1].Members, f.chats[1].OwnerOnly = 2, false
	f.fire(Event{ChatChanged: group})
	ev := h.event("chat_changed")
	if ev["chat"] != group || ev["members"] != float64(2) || ev["owner_only"] != false {
		t.Fatalf("chat_changed = %v", ev)
	}
	f.fire(Event{ChatChanged: never}) // never-listed: no event
	h.noEvent("chat_changed", 150*time.Millisecond)
}

// ---- inbound ---------------------------------------------------------------------

func TestInboundEmitsOwnerMessage(t *testing.T) {
	f := newFake()
	h := start(t, f, allowFlags...)
	h.hello()
	now := h.clock.Now()
	f.fire(Event{Message: &Inbound{
		Chat: group, MsgID: "M1", SenderUser: ownerLID, SenderDevice: 0, IsFromMe: true,
		Timestamp: now, Text: "find me 20 jobs",
		Quoted: &Quoted{MsgID: "Q1", Text: "earlier", Participant: "222@s.whatsapp.net"},
	}})
	ev := h.event("message")
	if ev["chat"] != group || ev["msg"] != "M1" || ev["text"] != "find me 20 jobs" || ev["sender_device"] != float64(0) ||
		ev["from_me"] != true || ev["replayed"] != false || ev["unsupported"] != "" || ev["sent_at"] != float64(now.Unix()) {
		t.Fatalf("message = %v", ev)
	}
	q := ev["quoted"].(map[string]any)
	if q["msg"] != "Q1" || q["from_owner"] != false || q["text"] != "earlier" {
		t.Fatalf("quoted = %v", q)
	}
	// Offline-sync window marks replayed.
	f.fire(Event{SyncStart: true})
	f.fire(Event{Message: &Inbound{Chat: group, MsgID: "M2", SenderUser: ownerPN, IsFromMe: true, Timestamp: now, Reaction: &Reaction{Emoji: "👍", TargetMsg: "X"}}})
	ev = h.event("message")
	if ev["replayed"] != true || ev["text"] != "" || ev["reaction"].(map[string]any)["target_msg"] != "X" {
		t.Fatalf("replayed reaction = %v", ev)
	}
	f.fire(Event{SyncDone: true})
}

func TestInboundDropsAndEchoBySentID(t *testing.T) {
	f := newFake()
	h := start(t, f, append(allowFlags, "--trusted-devices", "0")...)
	h.hello()
	now := h.clock.Now()
	sent := wantOK(t, h.call(1, "send_text", map[string]any{"chat": group, "text": "x"}))["msg"].(string)
	drops := []*Inbound{
		{Chat: never, MsgID: "D1", SenderUser: ownerPN, IsFromMe: true, Timestamp: now, Text: "x"},
		{Chat: stranger, MsgID: "D2", SenderUser: ownerPN, IsFromMe: true, Timestamp: now, Text: "x"},
		{Chat: group, MsgID: "D3", SenderUser: "222@s.whatsapp.net", Timestamp: now, Text: "x"},
		{Chat: group, MsgID: "D4", SenderUser: ownerPN, SenderDevice: 3, IsFromMe: true, Timestamp: now, Text: "x"},
		{Chat: group, MsgID: "D5", SenderUser: ownerPN, SenderDevice: 7, IsFromMe: true, Timestamp: now, Text: "x"},
		{Chat: group, MsgID: sent, SenderUser: ownerPN, IsFromMe: true, Timestamp: now, Text: "x"},
		{Chat: group, MsgID: "D7", SenderUser: ownerPN, IsFromMe: true, Timestamp: now, Text: "Mac · Done ✅"},
		{Chat: group, MsgID: "D8", SenderUser: ownerPN, IsFromMe: true, Timestamp: now.Add(-8 * 24 * time.Hour), Text: "x"},
	}
	for _, m := range drops {
		f.fire(Event{Message: m})
	}
	h.noEvent("message", 150*time.Millisecond)
	st := wantOK(t, h.call(2, "status", nil))
	dropped := st["dropped"].(map[string]any)
	for _, r := range []string{dropDenied, dropNotAllowed, dropNotOwner, dropUntrusted, dropEchoDevice, dropEchoID, dropEchoPrefix, dropTooOld} {
		if dropped[r] != float64(1) {
			t.Fatalf("dropped[%s] = %v (all: %v)", r, dropped[r], dropped)
		}
	}
}

func TestEchoIDSurvivesRestart(t *testing.T) {
	f := newFake()
	h := start(t, f, allowFlags...)
	h.hello()
	sent := wantOK(t, h.call(1, "send_text", map[string]any{"chat": group, "text": "x"}))["msg"].(string)
	h.in.Close()
	h.exitCode(2 * time.Second)
	r := loadSentRing(h.store + ".sent.json")
	if !r.has(sent) {
		t.Fatal("sent id not persisted")
	}
	b, _ := os.ReadFile(h.store + ".sent.json")
	if bytes.Contains(b, []byte(`"x"`)) {
		t.Fatal("sent log holds message text")
	}
	if st, _ := os.Stat(h.store + ".sent.json"); st.Mode().Perm() != 0o600 {
		t.Fatalf("sent log mode %v", st.Mode().Perm())
	}
}

func TestMediaDownloadPolicy(t *testing.T) {
	f := newFake()
	f.media["img"] = []byte("JPEGDATA")
	f.media["doc"] = []byte("%PDF")
	h := start(t, f, allowFlags...)
	h.hello()
	now := h.clock.Now()
	owner := func(id string, m *MediaIn) *Inbound {
		return &Inbound{Chat: group, MsgID: id, SenderUser: ownerPN, IsFromMe: true, Timestamp: now, Media: m}
	}
	// Unsupported kinds: flagged, never downloaded, no file.
	for _, k := range []string{"voice", "audio", "video", "sticker"} {
		f.fire(Event{Message: owner("U-"+k, &MediaIn{Kind: k, Handle: "img", Size: 10})})
		ev := h.event("message")
		if ev["unsupported"] != k || len(ev["attachments"].([]any)) != 0 {
			t.Fatalf("%s: %v", k, ev)
		}
	}
	// Non-owner image in the allowed chat: dropped before any download.
	f.fire(Event{Message: &Inbound{Chat: group, MsgID: "N1", SenderUser: "222@s.whatsapp.net", Timestamp: now, Media: &MediaIn{Kind: "image", Handle: "img"}}})
	h.noEvent("message", 100*time.Millisecond)
	f.mu.Lock()
	dl := f.downloads
	f.mu.Unlock()
	if dl != 0 {
		t.Fatalf("downloads = %d, want 0", dl)
	}
	if entries, _ := os.ReadDir(h.media); len(entries) != 0 {
		t.Fatalf("media dir not empty: %v", entries)
	}
	// Owner document: downloaded to --media-dir, 0600.
	f.fire(Event{Message: owner("DOC1", &MediaIn{Kind: "document", Handle: "doc", Mime: "application/pdf", FileName: "../my cv.pdf", Size: 4})})
	ev := h.event("message")
	att := ev["attachments"].([]any)[0].(map[string]any)
	path := att["path"].(string)
	if filepath.Dir(path) != h.media || att["name"] != "../my cv.pdf" || att["size"] != float64(4) || att["kind"] != "document" {
		t.Fatalf("attachment = %v", att)
	}
	st, err := os.Stat(path)
	if err != nil || st.Mode().Perm() != 0o600 {
		t.Fatalf("media file: %v %v", st, err)
	}
	if ds, _ := os.Stat(h.media); ds.Mode().Perm() != 0o700 {
		t.Fatalf("media dir mode %v", ds.Mode().Perm())
	}
	f.fire(Event{Message: owner("IMG1", &MediaIn{Kind: "image", Handle: "img", Mime: "image/jpeg", Size: 8})})
	if a := h.event("message")["attachments"].([]any); len(a) != 1 || a[0].(map[string]any)["path"] == "" {
		t.Fatalf("image attachments = %v", a)
	}
}

// ---- lifeline, logging, lock ---------------------------------------------------------

func TestStdinEOFExits0(t *testing.T) {
	h := start(t, newFake(), allowFlags...)
	h.hello()
	t0 := time.Now()
	h.in.Close()
	if c := h.exitCode(2 * time.Second); c != ExitOK {
		t.Fatalf("exit %d, want 0", c)
	}
	if d := time.Since(t0); d > 2*time.Second {
		t.Fatalf("took %v", d)
	}
}

func TestNoMessageTextOnStderr(t *testing.T) {
	f := newFake()
	f.chats = []ChatInfo{{ID: group, Subject: "SecretSubjectZ", Members: 1, OwnerOnly: true}}
	f.media["doc"] = []byte("x")
	h := start(t, f, allowFlags...)
	h.hello()
	now := h.clock.Now()
	file := filepath.Join(t.TempDir(), "SecretFileNameQ.pdf")
	os.WriteFile(file, []byte("x"), 0o600)
	msg := wantOK(t, h.call(1, "send_text", map[string]any{"chat": group, "text": "SecretOutboundA"}))["msg"].(string)
	wantOK(t, h.call(2, "edit", map[string]any{"chat": group, "msg": msg, "text": "SecretEditB"}))
	wantOK(t, h.call(3, "send_media", map[string]any{"chat": group, "path": file, "kind": "document", "caption": "SecretCaptionC", "file_name": "SecretNameD.pdf"}))
	wantOK(t, h.call(4, "list_chats", nil))
	wantOK(t, h.call(5, "react", map[string]any{"chat": group, "msg": msg, "from_me": true, "emoji": "👀"}))
	h.call(6, "SecretOpE", nil)
	h.sendRaw("SecretFrameF\n")
	h.nullResp()
	f.fire(Event{Message: &Inbound{Chat: group, MsgID: "I1", SenderUser: ownerPN, IsFromMe: true, Timestamp: now, Text: "SecretInboundG",
		Quoted: &Quoted{MsgID: "Q", Text: "SecretQuotedH"}}})
	h.event("message")
	f.fire(Event{Message: &Inbound{Chat: group, MsgID: "I2", SenderUser: ownerPN, IsFromMe: true, Timestamp: now, Text: "SecretCaptionI",
		Media: &MediaIn{Kind: "document", Handle: "doc", FileName: "SecretInFileJ.pdf"}}})
	h.event("message")
	f.fire(Event{Message: &Inbound{Chat: group, MsgID: "I3", SenderUser: "222@s.whatsapp.net", Timestamp: now, Text: "SecretOtherK"}})
	f.fire(Event{ChatChanged: group})
	h.event("chat_changed")
	h.in.Close()
	h.exitCode(2 * time.Second)
	logs := h.stderr.String()
	if !strings.Contains(logs, "send_text") || !strings.Contains(logs, "dropped{not_owner}") {
		t.Fatalf("stderr unexpectedly quiet:\n%s", logs)
	}
	for _, s := range []string{"Secret", "SecretOutboundA", "cv", "Interview"} {
		if strings.Contains(logs, s) {
			t.Fatalf("stderr contains %q:\n%s", s, logs)
		}
	}
}

func TestLockHeldExits75WithoutOpeningStore(t *testing.T) {
	dir := t.TempDir()
	store := filepath.Join(dir, "device.db")
	os.WriteFile(store, []byte("original"), 0o600)
	lock, err := lockStore(store)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	before, _ := os.Stat(store)
	opened := false
	var stderr bytes.Buffer
	code := Run([]string{"--store", store, "--media-dir", filepath.Join(dir, "m")}, strings.NewReader(""), io.Discard, &stderr,
		Options{Open: func(Config, *Logger) (Client, error) { opened = true; return newFake(), nil }})
	if code != ExitLocked {
		t.Fatalf("exit %d, want 75", code)
	}
	if opened {
		t.Fatal("store opened while locked")
	}
	after, _ := os.Stat(store)
	b, _ := os.ReadFile(store)
	if string(b) != "original" || !after.ModTime().Equal(before.ModTime()) {
		t.Fatal("store touched")
	}
	if _, err := os.Stat(filepath.Join(dir, "m")); err == nil {
		t.Fatal("media dir created by the locked-out instance")
	}
}

// Real processes: a second bridge on the same store exits 75, and a bridge
// whose stdin closes exits 0 within 2 s.
func TestProcessLevelLockAndLifeline(t *testing.T) {
	dir := t.TempDir()
	store := filepath.Join(dir, "device.db")
	args := []string{"-test.run=^$", "--", "--store", store, "--media-dir", filepath.Join(dir, "m"), "--allow", group, "--deny", never}

	a := exec.Command(os.Args[0], args...)
	a.Env = append(os.Environ(), "WA_BRIDGE_HELPER=1")
	stdinA, _ := a.StdinPipe()
	stdoutA, _ := a.StdoutPipe()
	if err := a.Start(); err != nil {
		t.Fatal(err)
	}
	// Wait for ready: A holds the lock.
	buf := make([]byte, 4096)
	n, _ := stdoutA.Read(buf)
	if !bytes.Contains(buf[:n], []byte(`"ready"`)) {
		t.Fatalf("A not ready: %q", buf[:n])
	}
	before, _ := os.Stat(store)

	b := exec.Command(os.Args[0], args...)
	b.Env = append(os.Environ(), "WA_BRIDGE_HELPER=1")
	err := b.Run()
	var ee *exec.ExitError
	if !errorsAs(err, &ee) || ee.ExitCode() != ExitLocked {
		t.Fatalf("second bridge: %v, want exit 75", err)
	}
	after, _ := os.Stat(store)
	if before != nil && after != nil && !after.ModTime().Equal(before.ModTime()) {
		t.Fatal("store touched by the second bridge")
	}

	t0 := time.Now()
	stdinA.Close()
	done := make(chan error, 1)
	go func() { done <- a.Wait() }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("A exit: %v", err)
		}
	case <-time.After(2 * time.Second):
		a.Process.Kill()
		t.Fatal("A did not exit within 2 s of stdin EOF")
	}
	t.Logf("lifeline exit in %v", time.Since(t0))
}

func errorsAs(err error, target **exec.ExitError) bool {
	e, ok := err.(*exec.ExitError)
	if ok {
		*target = e
	}
	return ok
}

// TestMain doubles as the helper process for process-level tests: it runs a
// bridge on the fake client (still no network).
func TestMain(m *testing.M) {
	if os.Getenv("WA_BRIDGE_HELPER") == "1" {
		args := os.Args[1:]
		for i, a := range args {
			if a == "--" {
				args = args[i+1:]
				break
			}
		}
		os.Exit(Run(args, os.Stdin, os.Stdout, os.Stderr, Options{Open: func(Config, *Logger) (Client, error) { return newFake(), nil }}))
	}
	os.Exit(m.Run())
}
