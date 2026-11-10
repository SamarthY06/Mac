package bridge

import (
	"strings"
	"time"
)

// EchoPrefix marks every message Mac writes. Inbound text starting with it is
// our own echo and is dropped (protocol §5 rule 4). chatd asserts it equals
// chatd.format.PREFIX.
const EchoPrefix = "Mac · "

// MaxAge drops anything older than this (protocol §5 rule 5).
const MaxAge = 7 * 24 * time.Hour

// filterState is what the inbound filter needs to decide. It is a snapshot,
// so filter itself is a pure function.
type filterState struct {
	allow    map[string]bool
	context  map[string]bool
	deny     map[string]bool
	trustAll bool
	trusted  map[int]bool
	self     Identity
	sent     func(id string) bool
	now      time.Time
}

// Drop reasons, as counted in status.dropped and logged as dropped{reason}.
const (
	dropDenied     = "denied"
	dropNotAllowed = "not_allowed"
	dropBroadcast  = "broadcast"
	dropStatus     = "status"
	dropProtocol   = "protocol"
	dropNotOwner   = "not_owner"
	dropUntrusted  = "untrusted_device"
	dropEchoDevice = "echo_device"
	dropEchoID     = "echo_id"
	dropEchoPrefix = "echo_prefix"
	dropTooOld     = "too_old"
	dropEmpty      = "empty"
)

// filter decides one inbound message (W10.1, protocol §5). It returns
// ("", false) for a `message` event (a trigger candidate), (reason, true) for
// a `context` event, and (reason, false) for a drop. The never-list is
// checked first, before anything else is looked at.
func filter(in *Inbound, s filterState) (reason string, context bool) {
	// Step 1.
	if s.deny[in.Chat] {
		return dropDenied, false
	}
	if !s.allow[in.Chat] {
		return dropNotAllowed, false
	}
	// Step 2: protocol, status, broadcast, age.
	switch {
	case in.IsStatus:
		return dropStatus, false
	case in.IsBroadcast:
		return dropBroadcast, false
	case in.IsProtocol:
		return dropProtocol, false
	case !in.Timestamp.IsZero() && s.now.Sub(in.Timestamp) > MaxAge:
		return dropTooOld, false
	}
	// Step 3: our own echo. The device check compares the full sender (our
	// account and this bridge's device id), so a member's device number that
	// equals ours never matches.
	owner := in.IsFromMe && isSelf(in.SenderUser, s.self)
	if isSelf(in.SenderUser, s.self) && in.SenderDevice == s.self.Device {
		return dropEchoDevice, false
	}
	if s.sent != nil && s.sent(in.MsgID) {
		return dropEchoID, false
	}
	if strings.HasPrefix(strings.TrimLeft(in.Text, " \t\r\n‎‏"), EchoPrefix) {
		return dropEchoPrefix, false
	}
	if in.Text == "" && in.Reaction == nil && in.Media == nil {
		return dropEmpty, false
	}
	// Step 4: the owner on a trusted device, else context (or nothing).
	reason = ""
	switch {
	case !owner:
		reason = dropNotOwner
	case !s.trustAll && !s.trusted[in.SenderDevice]:
		reason = dropUntrusted
	}
	if reason != "" {
		return reason, s.context[in.Chat] && in.Reaction == nil
	}
	return "", false // step 5
}

// isSelf reports whether a device-less user JID is the paired account,
// by phone-number JID or LID.
func isSelf(user string, self Identity) bool {
	if user == "" {
		return false
	}
	return (self.JID != "" && user == self.JID) || (self.LID != "" && user == self.LID)
}
