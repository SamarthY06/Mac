package bridge

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Protocol 2 (ARCHITECTURE W10, W7.7 bridge tests).

const member = "222@s.whatsapp.net"

var ctxFlags = []string{"--allow", group + "," + stranger, "--context", group, "--deny", never}

func memberMsg(now time.Time, id, text string) *Inbound {
	return &Inbound{Chat: group, MsgID: id, SenderUser: member, SenderName: "Priya Secretname", Timestamp: now, Text: text}
}

func TestContextFlagMustBeSubsetOfAllow(t *testing.T) {
	h := start(t, newFake(), "--allow", group, "--context", stranger)
	if c := h.exitCode(3 * time.Second); c != ExitBadArgs {
		t.Fatalf("exit %d, want 2", c)
	}
}

func TestContextEventsSplit(t *testing.T) {
	f := newFake()
	f.media["img"] = []byte("member picture")
	h := start(t, f, ctxFlags...)
	d := h.hello()
	if c := d["context"].([]any); len(c) != 1 || c[0] != group {
		t.Fatalf("hello context = %v", d["context"])
	}
	now := h.clock.Now()
	f.fire(Event{Message: memberMsg(now, "P1", "can you look at this? SecretText1")})
	ev := h.event("context")
	if ev["chat"] != group || ev["msg"] != "P1" || ev["sender"] != member || ev["sender_name"] != "Priya Secretname" ||
		ev["from_owner"] != false || ev["history"] != false || ev["text"] != "can you look at this? SecretText1" {
		t.Fatalf("context = %v", ev)
	}
	// A member whose device number equals ours is still context, never an echo.
	m := memberMsg(now, "P2", "hi")
	m.SenderDevice = 7
	f.fire(Event{Message: m})
	if ev := h.event("context"); ev["msg"] != "P2" {
		t.Fatalf("context = %v", ev)
	}
	// Context media: kind (and a document's name) only, never downloaded.
	m = memberMsg(now, "P3", "")
	m.Media = &MediaIn{Kind: "image", Mime: "image/jpeg", Size: 14, Handle: "img"}
	f.fire(Event{Message: m})
	ev = h.event("context")
	if media := ev["media"].(map[string]any); media["kind"] != "image" || len(media) != 1 {
		t.Fatalf("context media = %v", ev["media"])
	}
	m = memberMsg(now, "P4", "")
	m.Media = &MediaIn{Kind: "document", FileName: "SecretFile.pdf", Handle: "img"}
	f.fire(Event{Message: m})
	if media := h.event("context")["media"].(map[string]any); media["name"] != "SecretFile.pdf" {
		t.Fatalf("document context media = %v", media)
	}
	// A long text is clipped to 4,000 characters.
	f.fire(Event{Message: memberMsg(now, "P5", strings.Repeat("é", 5000))})
	if txt := h.event("context")["text"].(string); len([]rune(txt)) != maxContextText {
		t.Fatalf("clipped to %d", len([]rune(txt)))
	}
	// A member in an allowed but non-context chat, or in Timetable, gives nothing.
	m = memberMsg(now, "P6", "x")
	m.Chat = stranger
	f.fire(Event{Message: m})
	m = memberMsg(now, "P7", "x")
	m.Chat = never
	f.fire(Event{Message: m})
	// "Mac, stop" from a member is context, never a message.
	f.fire(Event{Message: memberMsg(now, "P8", "Mac, stop")})
	if ev := h.event("context"); ev["msg"] != "P8" {
		t.Fatalf("context = %v", ev)
	}
	h.noEvent("message", 150*time.Millisecond)
	h.noEvent("context", 50*time.Millisecond)
	if f.downloads != 0 {
		t.Fatalf("context media downloaded %d times", f.downloads)
	}
	// The owner's own message is still a trigger (trusted devices default: all).
	f.fire(Event{Message: &Inbound{Chat: group, MsgID: "O1", SenderUser: ownerPN, SenderDevice: 3, IsFromMe: true, Timestamp: now, Text: "Mac, go"}})
	if ev := h.event("message"); ev["msg"] != "O1" {
		t.Fatalf("message = %v", ev)
	}
	for _, s := range []string{"SecretText1", "Secretname", "SecretFile", "Mac, stop"} {
		if strings.Contains(h.stderr.String(), s) {
			t.Fatalf("stderr holds %q", s)
		}
	}
}

func TestSetAllowContext(t *testing.T) {
	h := start(t, newFake(), ctxFlags...)
	h.hello()
	wantErr(t, h.call(1, "set_allow", map[string]any{"allow": []string{group}, "context": []string{stranger}}), "bad_args")
	d := wantOK(t, h.call(3, "set_allow", map[string]any{"allow": []string{group, stranger}, "context": []string{stranger}}))
	if c := d["context"].([]any); len(c) != 1 || c[0] != stranger {
		t.Fatalf("context = %v", d)
	}
	// Omitted context: kept, minus chats no longer allowed.
	d = wantOK(t, h.call(4, "set_allow", map[string]any{"allow": []string{group}}))
	if c := d["context"].([]any); len(c) != 0 {
		t.Fatalf("context = %v", d)
	}
	wantErr(t, h.call(5, "set_allow", map[string]any{"allow": []string{group, never}, "context": []string{never}}), "denied")
}

func TestMembersOnlyForAllowedChats(t *testing.T) {
	f := newFake()
	f.members[group] = []Member{{JID: member, Number: "222", Name: "Priya"}, {JID: ownerPN, Number: "111", Name: ""}}
	h := start(t, f, ctxFlags...)
	h.hello()
	d := wantOK(t, h.call(1, "members", map[string]any{"chat": group}))
	if l := d["members"].([]any); len(l) != 2 || l[0].(map[string]any)["number"] != "222" {
		t.Fatalf("members = %v", d)
	}
	wantErr(t, h.call(2, "members", map[string]any{"chat": never}), "denied")
	wantErr(t, h.call(3, "members", map[string]any{"chat": "120363000000000099@g.us"}), "not_allowed")
	if strings.Contains(h.stderr.String(), "Priya") {
		t.Fatal("a member name reached stderr")
	}
}

func TestMentionsFiltered(t *testing.T) {
	f := newFake()
	f.members[group] = []Member{{JID: member, Number: "222", Name: "Priya"}, {JID: "333@lid", Number: "333", Name: "Arjun"}}
	h := start(t, f, ctxFlags...)
	h.hello()
	text := "@222 and @3334 look at this"
	d := wantOK(t, h.call(1, "send_text", map[string]any{"chat": group, "text": text,
		"mentions": []string{member, "333@lid", "444@s.whatsapp.net", member}}))
	if m := d["mentioned"].([]any); len(m) != 1 || m[0] != member {
		t.Fatalf("mentioned = %v", d["mentioned"])
	}
	msg := d["msg"].(string)
	d = wantOK(t, h.call(2, "edit", map[string]any{"chat": group, "msg": msg, "text": "@333 done", "mentions": []string{"333@lid"}}))
	if m := d["mentioned"].([]any); len(m) != 1 || m[0] != "333@lid" {
		t.Fatalf("edit mentioned = %v", d)
	}
	d = wantOK(t, h.call(3, "send_text", map[string]any{"chat": group, "text": "no tags"}))
	if m := d["mentioned"].([]any); len(m) != 0 {
		t.Fatalf("mentioned = %v", d)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.sendMent) != 3 || len(f.sendMent[0]) != 1 || f.sendMent[0][0] != member || len(f.sendMent[2]) != 0 {
		t.Fatalf("client mentions = %v", f.sendMent)
	}
	if !mentionIn("hi @222.", "222") || mentionIn("hi @2223", "222") || mentionIn("hi 222", "222") {
		t.Fatal("mentionIn")
	}
}

func TestQuotedSameChatAndOtherChat(t *testing.T) {
	f := newFake()
	f.media["qdoc"] = []byte("%PDF quoted")
	h := start(t, f, ctxFlags...)
	h.hello()
	now := h.clock.Now()
	// The quoted member message was seen earlier (as context), so its time is known.
	f.fire(Event{Message: memberMsg(now.Add(-time.Minute), "Q1", "see https://example.com/job.")})
	h.event("context")
	trigger := func(id string, q *Quoted) map[string]any {
		f.fire(Event{Message: &Inbound{Chat: group, MsgID: id, SenderUser: ownerPN, IsFromMe: true, Timestamp: now, Text: "Mac, what about this?", Quoted: q}})
		return h.event("message")["quoted"].(map[string]any)
	}
	q := trigger("T1", &Quoted{MsgID: "Q1", Text: "see https://example.com/job.", Participant: member, SenderName: "Priya",
		Links: []string{"https://example.com/job", "ftp://x"}, Media: &MediaIn{Kind: "document", FileName: "jd.pdf", Mime: "application/pdf", Size: 11, Handle: "qdoc"}})
	if q["msg"] != "Q1" || q["sender"] != member || q["sender_name"] != "Priya" || q["from_owner"] != false || q["from_mac"] != false ||
		q["text"] != "see https://example.com/job." || q["at"] != float64(now.Add(-time.Minute).Unix()) {
		t.Fatalf("quoted = %v", q)
	}
	if l := q["links"].([]any); len(l) != 1 || l[0] != "https://example.com/job" {
		t.Fatalf("links = %v", q["links"])
	}
	media := q["media"].(map[string]any)
	path, _ := media["path"].(string)
	if media["kind"] != "document" || path == "" || media["size"] != float64(11) {
		t.Fatalf("media = %v", media)
	}
	real, _ := filepath.EvalSymlinks(h.media)
	if filepath.Dir(path) != real {
		t.Fatalf("path %s outside %s", path, real)
	}
	if b, _ := os.ReadFile(path); !bytes.Equal(b, []byte("%PDF quoted")) {
		t.Fatal("quoted file content")
	}
	// Another chat (reply privately, forwarded) and Timetable: other_chat, no download.
	for i, remote := range []string{never, stranger, "999999@s.whatsapp.net"} {
		before := f.downloads
		q = trigger("T2"+string(rune('a'+i)), &Quoted{MsgID: "QX", Text: "SecretTimetable", RemoteJID: remote, Participant: member,
			Media: &MediaIn{Kind: "document", FileName: "tt.pdf", Size: 5, Handle: "qdoc"}})
		if len(q) != 2 || q["kind"] != "other_chat" || q["msg"] != "QX" || f.downloads != before {
			t.Fatalf("remote %s: quoted = %v", remote, q)
		}
	}
	// The same chat named explicitly is fine; Mac's own message is from_mac.
	sent := wantOK(t, h.call(1, "send_text", map[string]any{"chat": group, "text": "Mac · Send it?"}))["msg"].(string)
	q = trigger("T3", &Quoted{MsgID: sent, Text: "Mac · Send it?", RemoteJID: group, Participant: ownerPN})
	if q["from_mac"] != true || q["from_owner"] != true {
		t.Fatalf("quoted = %v", q)
	}
	// A member message quoting a document is context: nothing downloaded.
	before := f.downloads
	m := memberMsg(now, "P9", "look")
	m.Quoted = &Quoted{MsgID: "Q1", Media: &MediaIn{Kind: "document", Handle: "qdoc"}}
	f.fire(Event{Message: m})
	if ev := h.event("context"); ev["quoted_msg"] != "Q1" || f.downloads != before {
		t.Fatalf("context quote = %v", ev)
	}
	if strings.Contains(h.stderr.String(), "SecretTimetable") || strings.Contains(h.stderr.String(), "jd.pdf") {
		t.Fatal("stderr holds quoted content")
	}
}

func TestQuotedMediaCutOffAt25MBRead(t *testing.T) {
	f := newFake()
	f.media["big"] = bytes.Repeat([]byte("x"), maxQuotedMedia+(1<<20)) // larger than declared
	h := start(t, f, ctxFlags...)
	h.hello()
	now := h.clock.Now()
	f.fire(Event{Message: &Inbound{Chat: group, MsgID: "T1", SenderUser: ownerPN, IsFromMe: true, Timestamp: now, Text: "Mac, read this",
		Quoted: &Quoted{MsgID: "Q1", Participant: member, Media: &MediaIn{Kind: "document", FileName: "big.pdf", Size: 1000, Handle: "big"}}}})
	media := h.event("message")["quoted"].(map[string]any)["media"].(map[string]any)
	if _, has := media["path"]; has {
		t.Fatalf("over-cap media has a path: %v", media)
	}
	f.mu.Lock()
	written := f.written
	f.mu.Unlock()
	if written > maxQuotedMedia+64 {
		t.Fatalf("wrote %d bytes, past the cap", written)
	}
	if files, _ := os.ReadDir(h.media); len(files) != 0 {
		t.Fatalf("partial file left: %v", files)
	}
	// Declared over the cap: not even started.
	before := f.downloads
	f.fire(Event{Message: &Inbound{Chat: group, MsgID: "T2", SenderUser: ownerPN, IsFromMe: true, Timestamp: now, Text: "Mac, read",
		Quoted: &Quoted{MsgID: "Q2", Participant: member, Media: &MediaIn{Kind: "document", Size: maxQuotedMedia + 1, Handle: "big"}}}})
	h.event("message")
	if f.downloads != before {
		t.Fatal("declared-too-large media was downloaded")
	}
}

func TestHistoryIsContextOnlyAndBounded(t *testing.T) {
	f := newFake()
	h := start(t, f, append(ctxFlags, "--budget", "2")...)
	h.hello()
	now := h.clock.Now()
	f.fire(Event{Message: &Inbound{Chat: group, MsgID: "T1", SenderUser: ownerPN, IsFromMe: true, Timestamp: now, Text: "Mac, summarise"}})
	h.event("message")
	// A blob without an open request is dropped.
	f.history = []*Inbound{memberMsg(now.Add(-time.Hour), "H0", "early")}
	f.fire(Event{HistoryNotice: "n0"})
	h.noEvent("context", 100*time.Millisecond)

	wantErr(t, h.call(1, "history", map[string]any{"chat": stranger, "before_msg": "T1", "count": 30}), "not_allowed")
	wantErr(t, h.call(2, "history", map[string]any{"chat": group, "before_msg": "T1", "count": 51}), "bad_args")
	wantErr(t, h.call(3, "history", map[string]any{"chat": group, "before_msg": "NOPE", "count": 30}), "not_found")
	d := wantOK(t, h.call(4, "history", map[string]any{"chat": group, "before_msg": "T1", "count": 30}))
	if d["requested"] != true || f.histReqs[0].ID != "T1" || !f.histReqs[0].FromMe {
		t.Fatalf("history = %v %v", d, f.histReqs)
	}
	wantErr(t, h.call(5, "history", map[string]any{"chat": group, "before_msg": "T1", "count": 30}), "rate")
	// The blob: your old "Mac, stop" is context (history), never a message;
	// another chat's message is dropped.
	old := &Inbound{Chat: group, MsgID: "H1", SenderUser: ownerPN, IsFromMe: true, Timestamp: now.Add(-2 * time.Hour), Text: "Mac, stop"}
	other := memberMsg(now.Add(-time.Hour), "H2", "SecretOther")
	other.Chat = stranger
	f.mu.Lock()
	f.history = []*Inbound{old, other, memberMsg(now.Add(-time.Hour), "H3", "earlier member text")}
	f.mu.Unlock()
	f.fire(Event{HistoryNotice: "n1"})
	ev := h.event("context")
	if ev["msg"] != "H1" || ev["history"] != true || ev["from_owner"] != true {
		t.Fatalf("history context = %v", ev)
	}
	if ev := h.event("context"); ev["msg"] != "H3" {
		t.Fatalf("history context = %v", ev)
	}
	h.noEvent("message", 100*time.Millisecond)
	h.noEvent("context", 50*time.Millisecond)
	// The request counted toward the budget: one send left, then budget.
	wantOK(t, h.call(6, "send_text", map[string]any{"chat": group, "text": "x"}))
	wantErr(t, h.call(7, "send_text", map[string]any{"chat": group, "text": "y"}), "budget")
}

func TestHistoryRefusedUnderDryRun(t *testing.T) {
	f := newFake()
	h := start(t, f, append(ctxFlags, "--dry-run")...)
	h.hello()
	f.fire(Event{Message: &Inbound{Chat: group, MsgID: "T1", SenderUser: ownerPN, IsFromMe: true, Timestamp: h.clock.Now(), Text: "Mac, hi"}})
	h.event("message")
	wantErr(t, h.call(1, "history", map[string]any{"chat": group, "before_msg": "T1", "count": 30}), "bad_args")
	if len(f.histReqs) != 0 {
		t.Fatal("history sent under dry run")
	}
}

func TestRevokedOnlyForAllowedChats(t *testing.T) {
	f := newFake()
	h := start(t, f, ctxFlags...)
	h.hello()
	f.fire(Event{Revoked: &Revoked{Chat: never, MsgID: "T1"}})
	f.fire(Event{Revoked: &Revoked{Chat: "120363000000000077@g.us", MsgID: "X1"}}) // not allowed
	f.fire(Event{Revoked: &Revoked{Chat: group, MsgID: "G1"}})
	ev := h.event("revoked")
	if ev["chat"] != group || ev["msg"] != "G1" || len(ev) != 3 { // ev, chat, msg
		t.Fatalf("revoked = %v", ev)
	}
	h.noEvent("revoked", 100*time.Millisecond)
}
