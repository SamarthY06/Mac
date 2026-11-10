package bridge

import (
	"strings"
	"time"
)

// EchoPrefix marks every message Mac writes. Inbound text starting with it is
// our own echo and is dropped (protocol §5 rule 4).
const EchoPrefix = "Mac · "

// MaxAge drops anything older than this (protocol §5 rule 5).
const MaxAge = 7 * 24 * time.Hour

// filterState is what the inbound filter needs to decide. It is a snapshot,
// so filter itself is a pure function.
type filterState struct {
	allow   map[string]bool
	deny    map[string]bool
	trusted map[int]bool
	self    Identity
	sent    func(id string) bool
	now     time.Time
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

// filter returns "" when the message may be emitted, else the drop reason.
// The never-list is checked first, before anything else is looked at.
func filter(in *Inbound, s filterState) string {
	if s.deny[in.Chat] {
		return dropDenied
	}
	if !s.allow[in.Chat] {
		return dropNotAllowed
	}
	// Rule 5 first among the rest: it needs no identity.
	if in.IsStatus {
		return dropStatus
	}
	if in.IsBroadcast {
		return dropBroadcast
	}
	if in.IsProtocol {
		return dropProtocol
	}
	// Rule 2: the owner, cross-checked against the paired JID and its LID.
	if !in.IsFromMe || !isSelf(in.SenderUser, s.self) {
		return dropNotOwner
	}
	// Rule 4a: never our own device (checked before the trusted list, so a
	// misconfigured --trusted-devices cannot let our echo through).
	if in.SenderDevice == s.self.Device {
		return dropEchoDevice
	}
	// Rule 3.
	if !s.trusted[in.SenderDevice] {
		return dropUntrusted
	}
	// Rule 4b, 4c.
	if s.sent != nil && s.sent(in.MsgID) {
		return dropEchoID
	}
	if strings.HasPrefix(strings.TrimLeft(in.Text, " \t\r\n‎‏"), EchoPrefix) {
		return dropEchoPrefix
	}
	if !in.Timestamp.IsZero() && s.now.Sub(in.Timestamp) > MaxAge {
		return dropTooOld
	}
	if in.Text == "" && in.Reaction == nil && in.Media == nil {
		return dropEmpty
	}
	return ""
}

// isSelf reports whether a device-less user JID is the paired account,
// by phone-number JID or LID.
func isSelf(user string, self Identity) bool {
	if user == "" {
		return false
	}
	return (self.JID != "" && user == self.JID) || (self.LID != "" && user == self.LID)
}
