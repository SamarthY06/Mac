package bridge

import (
	"testing"
	"time"
)

const (
	group    = "120363000000000001@g.us" // "Interview Prep"
	never    = "120363000000000002@g.us" // "Timetable"
	stranger = "120363000000000003@g.us"
	ownerPN  = "111@s.whatsapp.net"
	ownerLID = "999@lid"
)

func baseState(now time.Time) filterState {
	return filterState{
		allow:   map[string]bool{group: true},
		deny:    map[string]bool{never: true},
		trusted: map[int]bool{0: true},
		self:    Identity{JID: ownerPN, LID: ownerLID, Device: 7},
		sent:    func(id string) bool { return id == "SENT1" },
		now:     now,
	}
}

func ownerMsg(now time.Time) *Inbound {
	return &Inbound{
		Chat: group, MsgID: "M1", SenderUser: ownerPN, SenderDevice: 0,
		IsFromMe: true, Timestamp: now.Add(-time.Minute), Text: "find me 20 jobs",
	}
}

func TestFilterRules(t *testing.T) {
	now := time.Date(2026, 11, 10, 12, 0, 0, 0, time.UTC)
	cases := []struct {
		name   string
		mutate func(*Inbound, *filterState)
		want   string
	}{
		{"owner message passes", func(*Inbound, *filterState) {}, ""},
		{"owner by LID passes", func(m *Inbound, _ *filterState) { m.SenderUser = ownerLID }, ""},
		{"rule1 never-listed chat dropped", func(m *Inbound, _ *filterState) { m.Chat = never }, dropDenied},
		{"rule1 never-list wins even if allowlisted", func(m *Inbound, s *filterState) { m.Chat = never; s.allow[never] = true }, dropDenied},
		{"rule1 never-list checked before everything", func(m *Inbound, _ *filterState) {
			m.Chat, m.IsFromMe, m.IsProtocol, m.SenderDevice = never, false, true, 3
		}, dropDenied},
		{"rule1 not allowlisted", func(m *Inbound, _ *filterState) { m.Chat = stranger }, dropNotAllowed},
		{"rule2 not from me", func(m *Inbound, _ *filterState) { m.IsFromMe = false; m.SenderUser = "222@s.whatsapp.net" }, dropNotOwner},
		{"rule2 from-me flag but other sender", func(m *Inbound, _ *filterState) { m.SenderUser = "222@s.whatsapp.net" }, dropNotOwner},
		{"rule2 unlinked bridge drops", func(_ *Inbound, s *filterState) { s.self = Identity{} }, dropNotOwner},
		{"rule3 untrusted device", func(m *Inbound, _ *filterState) { m.SenderDevice = 3 }, dropUntrusted},
		{"rule3 extra trusted device passes", func(m *Inbound, s *filterState) { m.SenderDevice = 3; s.trusted[3] = true }, ""},
		{"rule4 echo by device", func(m *Inbound, _ *filterState) { m.SenderDevice = 7 }, dropEchoDevice},
		{"rule4 echo by device even if trusted", func(m *Inbound, s *filterState) { m.SenderDevice = 7; s.trusted[7] = true }, dropEchoDevice},
		{"rule4 echo by sent id", func(m *Inbound, _ *filterState) { m.MsgID = "SENT1" }, dropEchoID},
		{"rule4 echo by prefix", func(m *Inbound, _ *filterState) { m.Text = "Mac · Done ✅" }, dropEchoPrefix},
		{"rule4 echo by prefix after spaces", func(m *Inbound, _ *filterState) { m.Text = "  Mac · On it." }, dropEchoPrefix},
		{"rule4 'Mac' without the dot passes", func(m *Inbound, _ *filterState) { m.Text = "Mac, find jobs" }, ""},
		{"rule5 protocol", func(m *Inbound, _ *filterState) { m.IsProtocol = true }, dropProtocol},
		{"rule5 status", func(m *Inbound, _ *filterState) { m.IsStatus = true }, dropStatus},
		{"rule5 broadcast", func(m *Inbound, _ *filterState) { m.IsBroadcast = true }, dropBroadcast},
		{"rule5 older than 7 days", func(m *Inbound, _ *filterState) { m.Timestamp = now.Add(-7*24*time.Hour - time.Second) }, dropTooOld},
		{"rule5 just under 7 days passes", func(m *Inbound, _ *filterState) { m.Timestamp = now.Add(-7*24*time.Hour + time.Minute) }, ""},
		{"empty message dropped", func(m *Inbound, _ *filterState) { m.Text = "" }, dropEmpty},
		{"reaction passes", func(m *Inbound, _ *filterState) { m.Text = ""; m.Reaction = &Reaction{Emoji: "👍", TargetMsg: "X"} }, ""},
		{"voice note passes as unsupported", func(m *Inbound, _ *filterState) { m.Text = ""; m.Media = &MediaIn{Kind: "voice"} }, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := ownerMsg(now)
			s := baseState(now)
			c.mutate(m, &s)
			if got, ctx := filter(m, s); got != c.want || ctx {
				t.Fatalf("filter = %q (context %v), want %q", got, ctx, c.want)
			}
		})
	}
}

// W10.1: the trigger/context split in a context chat.
func TestFilterContextSplit(t *testing.T) {
	now := time.Date(2026, 11, 10, 12, 0, 0, 0, time.UTC)
	member := func() *Inbound {
		return &Inbound{Chat: group, MsgID: "P1", SenderUser: "222@s.whatsapp.net", SenderDevice: 0,
			Timestamp: now.Add(-time.Minute), Text: "can you look at this?"}
	}
	cases := []struct {
		name    string
		mutate  func(*Inbound, *filterState)
		want    string
		context bool
	}{
		{"member in a context chat is context", func(*Inbound, *filterState) {}, dropNotOwner, true},
		{"member in a non-context chat is dropped", func(_ *Inbound, s *filterState) { s.context = map[string]bool{} }, dropNotOwner, false},
		{"member whose device equals ours is still context", func(m *Inbound, _ *filterState) { m.SenderDevice = 7 }, dropNotOwner, true},
		{"member in a non-allowed chat is dropped", func(m *Inbound, s *filterState) { m.Chat = stranger; s.context[stranger] = true }, dropNotAllowed, false},
		{"member in Timetable is dropped", func(m *Inbound, s *filterState) { m.Chat = never; s.context[never] = true }, dropDenied, false},
		{"member text with our prefix is an echo", func(m *Inbound, _ *filterState) { m.Text = "Mac · hi" }, dropEchoPrefix, false},
		{"member reaction is dropped", func(m *Inbound, _ *filterState) { m.Text = ""; m.Reaction = &Reaction{Emoji: "👍"} }, dropNotOwner, false},
		{"member media is context", func(m *Inbound, _ *filterState) { m.Text = ""; m.Media = &MediaIn{Kind: "image"} }, dropNotOwner, true},
		{"old member message dropped", func(m *Inbound, _ *filterState) { m.Timestamp = now.Add(-8 * 24 * time.Hour) }, dropTooOld, false},
		{"owner on an untrusted device is context", func(m *Inbound, _ *filterState) {
			m.SenderUser, m.IsFromMe, m.SenderDevice = ownerPN, true, 3
		}, dropUntrusted, true},
		{"owner on any device with trust all is a trigger", func(m *Inbound, s *filterState) {
			m.SenderUser, m.IsFromMe, m.SenderDevice, s.trustAll = ownerPN, true, 3, true
		}, "", false},
		{"our own device stays an echo with trust all", func(m *Inbound, s *filterState) {
			m.SenderUser, m.IsFromMe, m.SenderDevice, s.trustAll = ownerLID, true, 7, true
		}, dropEchoDevice, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := member()
			s := baseState(now)
			s.context = map[string]bool{group: true}
			c.mutate(m, &s)
			if got, ctx := filter(m, s); got != c.want || ctx != c.context {
				t.Fatalf("filter = %q (context %v), want %q (context %v)", got, ctx, c.want, c.context)
			}
		})
	}
}
