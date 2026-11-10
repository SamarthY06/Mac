package wa

import (
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"go.mau.fi/whatsmeow/proto/waCommon"
	"go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
	"google.golang.org/protobuf/proto"

	"github.com/SamarthY06/Mac/bridges/whatsapp/internal/bridge"
)

// These tests open a real (temp) SQLite store through whatsmeow but never
// call Connect: no network is touched.
func openTest(t *testing.T) *Client {
	t.Helper()
	dir := t.TempDir()
	cfg := bridge.Config{Store: filepath.Join(dir, "wa", "device.db"), MediaDir: filepath.Join(dir, "m")}
	cl, err := Open(cfg, bridge.NewLogger(io.Discard))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(cl.Close)
	c := cl.(*Client)
	if c.Linked() || c.Connected() || c.Identity() != (bridge.Identity{}) {
		t.Fatal("fresh store should be unlinked")
	}
	st, _ := os.Stat(cfg.Store)
	if st.Mode().Perm() != 0o600 {
		t.Fatalf("store mode %v", st.Mode().Perm())
	}
	// Pretend a device is paired (in memory only).
	c.cli.Store.ID = &types.JID{User: "111", Device: 7, Server: types.DefaultUserServer}
	c.cli.Store.LID = types.JID{User: "999", Device: 7, Server: types.HiddenUserServer}
	return c
}

func msgEvent(chat, sender types.JID, fromMe bool, m *waE2E.Message) *events.Message {
	return &events.Message{
		Info: types.MessageInfo{
			MessageSource: types.MessageSource{Chat: chat, Sender: sender, IsFromMe: fromMe, IsGroup: chat.Server == types.GroupServer},
			ID:            "3EB0AA", Timestamp: time.Unix(1762790400, 0),
		},
		Message: m,
	}
}

func TestConvert(t *testing.T) {
	c := openTest(t)
	if id := c.Identity(); id.JID != "111@s.whatsapp.net" || id.LID != "999@lid" || id.Device != 7 || !c.Linked() {
		t.Fatalf("identity %+v", id)
	}
	group := types.JID{User: "120363000000000001", Server: types.GroupServer}
	phone := types.JID{User: "999", Device: 0, Server: types.HiddenUserServer} // owner's phone by LID

	in := c.convert(msgEvent(group, phone, true, &waE2E.Message{Conversation: proto.String("hi")}))
	if in.Chat != group.String() || in.SenderUser != "999@lid" || in.SenderDevice != 0 || !in.IsFromMe || in.Text != "hi" || in.IsProtocol {
		t.Fatalf("text: %+v", in)
	}

	kinds := map[string]*waE2E.Message{
		"voice":   {AudioMessage: &waE2E.AudioMessage{PTT: proto.Bool(true)}},
		"audio":   {AudioMessage: &waE2E.AudioMessage{}},
		"video":   {VideoMessage: &waE2E.VideoMessage{Caption: proto.String("c")}},
		"sticker": {StickerMessage: &waE2E.StickerMessage{}},
	}
	for k, m := range kinds {
		in := c.convert(msgEvent(group, phone, true, m))
		if in.Media == nil || in.Media.Kind != k || in.Media.Handle != nil {
			t.Fatalf("%s: %+v (must carry no download handle)", k, in.Media)
		}
	}
	img := c.convert(msgEvent(group, phone, true, &waE2E.Message{ImageMessage: &waE2E.ImageMessage{Caption: proto.String("look"), Mimetype: proto.String("image/jpeg"), FileLength: proto.Uint64(5)}}))
	if img.Media.Kind != "image" || img.Media.Handle == nil || img.Text != "look" || img.Media.Size != 5 {
		t.Fatalf("image: %+v", img)
	}
	doc := c.convert(msgEvent(group, phone, true, &waE2E.Message{DocumentMessage: &waE2E.DocumentMessage{FileName: proto.String("cv.pdf"), Mimetype: proto.String("application/pdf")}}))
	if doc.Media.Kind != "document" || doc.Media.FileName != "cv.pdf" {
		t.Fatalf("doc: %+v", doc.Media)
	}
	re := c.convert(msgEvent(group, phone, true, &waE2E.Message{ReactionMessage: &waE2E.ReactionMessage{Text: proto.String("👍"), Key: &waCommon.MessageKey{ID: proto.String("T1")}}}))
	if re.Reaction == nil || re.Reaction.Emoji != "👍" || re.Reaction.TargetMsg != "T1" {
		t.Fatalf("reaction: %+v", re)
	}
	quoted := c.convert(msgEvent(group, phone, true, &waE2E.Message{ExtendedTextMessage: &waE2E.ExtendedTextMessage{
		Text: proto.String("this one"),
		ContextInfo: &waE2E.ContextInfo{StanzaID: proto.String("Q1"), Participant: proto.String("999:3@lid"),
			QuotedMessage: &waE2E.Message{Conversation: proto.String("old")}},
	}}))
	if quoted.Quoted == nil || quoted.Quoted.MsgID != "Q1" || quoted.Quoted.Text != "old" || quoted.Quoted.Participant != "111@s.whatsapp.net" {
		t.Fatalf("quoted: %+v", quoted.Quoted)
	}
	edit := msgEvent(group, phone, true, &waE2E.Message{Conversation: proto.String("x")})
	edit.IsEdit = true
	if !c.convert(edit).IsProtocol {
		t.Fatal("edit should be protocol")
	}
	if !c.convert(msgEvent(group, phone, true, &waE2E.Message{ProtocolMessage: &waE2E.ProtocolMessage{}})).IsProtocol {
		t.Fatal("protocol message")
	}
	st := c.convert(msgEvent(types.StatusBroadcastJID, phone, true, &waE2E.Message{Conversation: proto.String("x")}))
	if !st.IsStatus || !st.IsBroadcast {
		t.Fatalf("status: %+v", st)
	}
	// "Message yourself" addressed by LID is reported by phone JID.
	self := c.convert(msgEvent(types.JID{User: "999", Server: types.HiddenUserServer}, phone, true, &waE2E.Message{Conversation: proto.String("x")}))
	if self.Chat != "111@s.whatsapp.net" {
		t.Fatalf("self chat = %s", self.Chat)
	}
}

func TestOwnerOnlyGroup(t *testing.T) {
	c := openTest(t)
	g := &types.GroupInfo{JID: types.JID{User: "1", Server: types.GroupServer}, GroupName: types.GroupName{Name: "Interview Prep"},
		Participants: []types.GroupParticipant{{JID: types.JID{User: "999", Server: types.HiddenUserServer}}}}
	if info := c.groupInfo(g); !info.OwnerOnly || info.Members != 1 || info.Subject != "Interview Prep" {
		t.Fatalf("owner-only: %+v", info)
	}
	g.Participants = append(g.Participants, types.GroupParticipant{JID: types.JID{User: "222", Server: types.DefaultUserServer}})
	if info := c.groupInfo(g); info.OwnerOnly || info.Members != 2 {
		t.Fatalf("shared: %+v", info)
	}
	if s := c.selfChat(); s.ID != "111@s.whatsapp.net" || !s.IsSelf || !s.OwnerOnly {
		t.Fatalf("self chat: %+v", s)
	}
}

func TestLogoutReason(t *testing.T) {
	cases := map[string]*events.LoggedOut{
		"phone":  {OnConnect: true, Reason: events.ConnectFailureLoggedOut},
		"banned": {OnConnect: true, Reason: events.ConnectFailureUnknownLogout},
		"other":  {OnConnect: true, Reason: events.ConnectFailureMainDeviceGone},
	}
	for want, ev := range cases {
		if got := logoutReason(ev); got != want {
			t.Errorf("%v: got %s want %s", ev.Reason, got, want)
		}
	}
	if logoutReason(&events.LoggedOut{OnConnect: false, Reason: events.ConnectFailureLoggedOut}) != "phone" {
		t.Error("device_removed should be phone")
	}
}

func TestConvertProtocol2(t *testing.T) {
	c := openTest(t)
	group := types.JID{User: "120363000000000001", Server: types.GroupServer}
	phone := types.JID{User: "999", Server: types.HiddenUserServer}
	ext := func(ci *waE2E.ContextInfo) *bridge.Quoted {
		return c.convert(msgEvent(group, phone, true, &waE2E.Message{ExtendedTextMessage: &waE2E.ExtendedTextMessage{
			Text: proto.String("Mac, this"), ContextInfo: ci}})).Quoted
	}
	// A same-chat quote of a member's document with a link preview.
	q := ext(&waE2E.ContextInfo{StanzaID: proto.String("Q1"), Participant: proto.String("222@s.whatsapp.net"),
		QuotedMessage: &waE2E.Message{DocumentMessage: &waE2E.DocumentMessage{FileName: proto.String("jd.pdf"),
			Mimetype: proto.String("application/pdf"), FileLength: proto.Uint64(80211), Caption: proto.String("see")}}})
	if q.RemoteJID != "" || q.Participant != "222@s.whatsapp.net" || q.Media == nil || q.Media.Kind != "document" ||
		q.Media.Handle == nil || q.Media.Size != 80211 || q.Text != "see" {
		t.Fatalf("quoted doc: %+v %+v", q, q.Media)
	}
	q = ext(&waE2E.ContextInfo{StanzaID: proto.String("Q2"), QuotedMessage: &waE2E.Message{ExtendedTextMessage: &waE2E.ExtendedTextMessage{
		Text: proto.String("look https://example.com/a"), MatchedText: proto.String("https://example.com/a")}}})
	if len(q.Links) != 1 || q.Links[0] != "https://example.com/a" || q.Media != nil {
		t.Fatalf("preview links: %+v", q)
	}
	// Reply privately / forwarded: the quote names another chat.
	q = ext(&waE2E.ContextInfo{StanzaID: proto.String("Q3"), RemoteJID: proto.String("120363000000000009@g.us"),
		QuotedMessage: &waE2E.Message{Conversation: proto.String("timetable")}})
	if q.RemoteJID != "120363000000000009@g.us" {
		t.Fatalf("remote: %+v", q)
	}
	// A quoted voice note has no handle.
	q = ext(&waE2E.ContextInfo{StanzaID: proto.String("Q4"), QuotedMessage: &waE2E.Message{AudioMessage: &waE2E.AudioMessage{PTT: proto.Bool(true)}}})
	if q.Media == nil || q.Media.Kind != "voice" || q.Media.Handle != nil {
		t.Fatalf("voice: %+v", q.Media)
	}
	// Push name is the fallback sender name (no contact stored).
	ev := msgEvent(group, types.JID{User: "222", Server: types.DefaultUserServer}, false, &waE2E.Message{Conversation: proto.String("hi")})
	ev.Info.PushName = "Priya"
	if in := c.convert(ev); in.SenderName != "Priya" {
		t.Fatalf("sender name %q", in.SenderName)
	}
}

func TestTextMessageMentions(t *testing.T) {
	if m := textMessage("hi", nil, nil); m.GetConversation() != "hi" {
		t.Fatalf("plain: %v", m)
	}
	m := textMessage("@222 hi", nil, []string{"222@s.whatsapp.net"})
	if x := m.GetExtendedTextMessage(); x.GetText() != "@222 hi" || len(x.GetContextInfo().GetMentionedJID()) != 1 {
		t.Fatalf("mention: %v", m)
	}
}

func TestOnlyOnDemandHistoryReachesTheBridge(t *testing.T) {
	c := openTest(t)
	var got []bridge.Event
	c.SetHandler(func(e bridge.Event) { got = append(got, e) })
	self := types.JID{User: "111", Server: types.DefaultUserServer}
	for _, typ := range []waE2E.HistorySyncType{waE2E.HistorySyncType_INITIAL_BOOTSTRAP, waE2E.HistorySyncType_RECENT, waE2E.HistorySyncType_ON_DEMAND} {
		c.onWAEvent(msgEvent(self, self, true, &waE2E.Message{ProtocolMessage: &waE2E.ProtocolMessage{
			HistorySyncNotification: &waE2E.HistorySyncNotification{SyncType: typ.Enum()}}}))
	}
	// From another sender: never.
	c.onWAEvent(msgEvent(self, types.JID{User: "222", Server: types.DefaultUserServer}, false, &waE2E.Message{ProtocolMessage: &waE2E.ProtocolMessage{
		HistorySyncNotification: &waE2E.HistorySyncNotification{SyncType: waE2E.HistorySyncType_ON_DEMAND.Enum()}}}))
	if len(got) != 1 || got[0].HistoryNotice == nil || got[0].Message != nil {
		t.Fatalf("events: %+v", got)
	}
}
