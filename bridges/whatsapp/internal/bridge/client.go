package bridge

import (
	"context"
	"errors"
	"io"
	"os"
	"time"
)

// Client is everything the bridge needs from WhatsApp. The production
// implementation wraps whatsmeow (internal/wa); tests use a fake. All JIDs are
// strings in whatsmeow's String() form; chat JIDs carry no device part.
type Client interface {
	// Linked reports whether the store holds a paired device.
	Linked() bool
	// Connected reports whether the websocket is up and logged in.
	Connected() bool
	// Identity of the paired device; zero value when unlinked.
	Identity() Identity

	// Connect starts connecting a linked device (non-blocking beyond the
	// dial). Events arrive through the handler.
	Connect() error
	// Disconnect closes the connection; whatsmeow auto-reconnect stops.
	Disconnect()
	// StartPair begins QR pairing of a fresh device and connects. The channel
	// yields codes and ends with a terminal item.
	StartPair(ctx context.Context) (<-chan QRItem, error)
	// Logout unlinks the device on the phone and deletes the keys. The client
	// stays usable for a later StartPair.
	Logout(ctx context.Context) error

	ListChats(ctx context.Context) ([]ChatInfo, error)
	GetChat(ctx context.Context, chat string) (ChatInfo, error)

	// Members lists a chat's current participants (GetGroupInfo; the self
	// chat lists only the owner). Names: contact name, else push name.
	Members(ctx context.Context, chat string) ([]Member, error)

	// SendText and Edit set ContextInfo.MentionedJID to mentions (already
	// checked by the bridge against the participants and the text).
	SendText(ctx context.Context, chat, text string, reply *ReplyRef, mentions []string) (msgID string, err error)
	Edit(ctx context.Context, chat, msgID, text string, mentions []string) error
	// React reacts to msgID sent by sender (a user JID). emoji "" removes it.
	React(ctx context.Context, chat, sender, msgID, emoji string) error
	SendMedia(ctx context.Context, m MediaOut) (msgID string, err error)
	SetPresence(ctx context.Context, available bool) error
	ChatPresence(ctx context.Context, chat string, composing bool) error

	// Download fetches inbound media. Only ever called for images and
	// documents of messages that passed the inbound filter.
	Download(ctx context.Context, m *MediaIn) ([]byte, error)
	// DownloadToFile streams media into f, which refuses writes past its cap
	// (quoted media: the limit is on the bytes actually read).
	DownloadToFile(ctx context.Context, m *MediaIn, f *CappedFile) error

	// RequestHistory sends an on-demand history request for the messages
	// before `before` to the owner's phone (a peer message). The answer
	// arrives later as an Event.HistoryNotice.
	RequestHistory(ctx context.Context, before MsgMeta, count int) error
	// FetchHistory downloads and parses an on-demand history blob. It never
	// downloads media.
	FetchHistory(ctx context.Context, notice any) ([]*Inbound, error)

	// SetHandler installs the event sink. Called once, before Connect.
	SetHandler(func(Event))
	// Close releases the store.
	Close()
}

// Identity of the paired device.
type Identity struct {
	JID    string // "<user>@s.whatsapp.net" (no device)
	LID    string // "<lid>@lid" (no device), may be empty
	Device int    // this bridge's device id
}

// QRItem is one step of pairing.
type QRItem struct {
	Code    string        // set for a new code
	Timeout time.Duration // validity of Code
	Done    string        // "" while codes flow; "success", "timeout" or "error"
	Err     string        // detail for "error" (no user content)
}

// ChatInfo is one entry of list_chats / chat_changed.
type ChatInfo struct {
	ID        string `json:"id"`
	Subject   string `json:"subject"`
	Members   int    `json:"members"`
	OwnerOnly bool   `json:"owner_only"`
	IsSelf    bool   `json:"is_self"`
}

// Member is one participant of a chat. Number is the user part of JID, the
// form an @-mention takes in the text.
type Member struct {
	JID    string `json:"jid"`
	Number string `json:"number"`
	Name   string `json:"name"`
}

// MsgMeta identifies a message for a history request.
type MsgMeta struct {
	Chat   string
	ID     string
	FromMe bool
	At     time.Time
}

// ErrTooLarge is returned by CappedFile once its cap is passed.
var ErrTooLarge = errors.New("media over the size cap")

// CappedFile is an *os.File that refuses to grow past Cap bytes, so a
// download is cut off on the bytes it actually writes (a retry that seeks
// back to 0 starts over).
type CappedFile struct {
	*os.File
	Cap int64
}

func (f *CappedFile) Write(p []byte) (int, error) {
	pos, err := f.File.Seek(0, io.SeekCurrent)
	if err != nil {
		return 0, err
	}
	if pos+int64(len(p)) > f.Cap {
		return 0, ErrTooLarge
	}
	return f.File.Write(p)
}

func (f *CappedFile) WriteAt(p []byte, off int64) (int, error) {
	if off+int64(len(p)) > f.Cap {
		return 0, ErrTooLarge
	}
	return f.File.WriteAt(p, off)
}

// ReplyRef is a resolved quote target.
type ReplyRef struct {
	MsgID  string
	Sender string // user JID of the quoted message's author
}

// MediaOut is an outbound file.
type MediaOut struct {
	Chat     string
	Path     string
	Data     []byte
	Kind     string // "document" or "image"
	Caption  string
	FileName string
	Mime     string
}

// MediaIn describes inbound media. Handle is opaque to the bridge.
type MediaIn struct {
	Kind     string // image, document, voice, audio, video, sticker
	Mime     string
	FileName string
	Size     int64
	Handle   any
}

// Inbound is a parsed incoming message, before the filter.
type Inbound struct {
	Chat         string // chat JID, no device
	MsgID        string
	SenderUser   string // sender JID without device, e.g. 123@s.whatsapp.net or 456@lid
	SenderAlt    string // alternate address of the sender (PN <-> LID), may be empty
	SenderDevice int
	SenderName   string // contact name, else push name; never logged
	IsFromMe     bool
	Timestamp    time.Time
	Replayed     bool

	IsProtocol  bool // protocol message, edit, revoke, key distribution...
	IsStatus    bool // status@broadcast
	IsBroadcast bool // broadcast list

	Text     string
	Quoted   *Quoted
	Reaction *Reaction
	Media    *MediaIn
}

// Quoted is the context of a reply, as parsed from ContextInfo.
type Quoted struct {
	MsgID       string
	Text        string
	Participant string   // author (user JID, no device); "" when unknown
	SenderName  string   // contact or push name of the author, when known
	RemoteJID   string   // ContextInfo.RemoteJID: set when the quote is from another chat
	Links       []string // URLs of the quoted text and its link preview
	Media       *MediaIn // image/document/voice/... of the quoted message
}

// Reaction is an inbound reaction.
type Reaction struct {
	Emoji     string `json:"emoji"`
	TargetMsg string `json:"target_msg"`
}

// Event is something the client reports. Exactly one field is set.
type Event struct {
	Connected   bool
	Offline     string // reason
	LoggedOut   string // phone|banned|other
	Message     *Inbound
	ChatChanged string // chat JID whose info changed
	SyncStart   bool   // offline sync began
	SyncDone    bool   // offline sync finished
	StoreError  string // detail
	// HistoryNotice is an on-demand history notification (opaque handle for
	// FetchHistory). Other history syncs are discarded by the client.
	HistoryNotice any
	// Revoked is a message deleted on the phone (for everyone, or for me).
	Revoked *Revoked
}

// Revoked names a deleted message.
type Revoked struct {
	Chat  string // chat JID, no device
	MsgID string
}
