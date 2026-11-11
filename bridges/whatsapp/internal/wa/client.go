// Package wa adapts whatsmeow to bridge.Client. It is the only package that
// talks to WhatsApp; unit tests never use it.
package wa

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/store"
	"go.mau.fi/whatsmeow/store/sqlstore"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
	waLog "go.mau.fi/whatsmeow/util/log"
	"google.golang.org/protobuf/proto"
	_ "modernc.org/sqlite" // pure-Go driver "sqlite", no cgo

	"github.com/SamarthY06/OpenMac/bridges/whatsapp/internal/bridge"
)

// DeviceName is what the phone shows under Linked devices (W5.6).
const DeviceName = "Mac (macOS)"

// Client implements bridge.Client with whatsmeow.
type Client struct {
	container *sqlstore.Container
	wlog      waLog.Logger
	log       *bridge.Logger

	mu      sync.Mutex
	cli     *whatsmeow.Client
	handler func(bridge.Event)
}

// Open opens (creating if needed) the store and returns a client. The caller
// already holds the store lock.
func Open(cfg bridge.Config, log *bridge.Logger) (bridge.Client, error) {
	if strings.ContainsAny(cfg.Store, "?#%") {
		return nil, errors.New("store path must not contain ? # or %")
	}
	dir := filepath.Dir(cfg.Store)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	_ = os.Chmod(dir, 0o700)
	f, err := os.OpenFile(cfg.Store, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, err
	}
	f.Close()
	_ = os.Chmod(cfg.Store, 0o600)

	store.SetOSInfo(DeviceName, [3]uint32{1, 0, 0})
	wlog := &logAdapter{log: log}
	dsn := "file:" + cfg.Store + "?_pragma=foreign_keys(1)&_pragma=busy_timeout(5000)"
	container, err := sqlstore.New(context.Background(), "sqlite", dsn, wlog)
	if err != nil {
		return nil, fmt.Errorf("open store: %w", err)
	}
	device, err := container.GetFirstDevice(context.Background())
	if err != nil {
		_ = container.Close()
		return nil, fmt.Errorf("load device: %w", err)
	}
	c := &Client{container: container, wlog: wlog, log: log}
	c.cli = c.newWA(device)
	return c, nil
}

func (c *Client) newWA(device *store.Device) *whatsmeow.Client {
	cli := whatsmeow.NewClient(device, c.wlog)
	// History blobs carry every chat's past messages; never fetch them.
	cli.ManualHistorySyncDownload = true
	cli.EnableAutoReconnect = true
	cli.AddEventHandler(c.onWAEvent)
	return cli
}

func (c *Client) wa() *whatsmeow.Client {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.cli
}

// reset swaps in a fresh, unpaired device after the keys were deleted.
func (c *Client) reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.cli != nil {
		c.cli.Disconnect()
		c.cli.RemoveEventHandlers()
	}
	c.cli = c.newWA(c.container.NewDevice())
}

func (c *Client) emit(e bridge.Event) {
	c.mu.Lock()
	h := c.handler
	c.mu.Unlock()
	if h != nil {
		h(e)
	}
}

func (c *Client) SetHandler(h func(bridge.Event)) {
	c.mu.Lock()
	c.handler = h
	c.mu.Unlock()
}

func (c *Client) Linked() bool {
	cli := c.wa()
	return cli.Store.ID != nil && !cli.Store.Deleted
}

func (c *Client) Connected() bool {
	cli := c.wa()
	return cli.IsConnected() && cli.IsLoggedIn()
}

func (c *Client) Identity() bridge.Identity {
	cli := c.wa()
	if cli.Store.ID == nil {
		return bridge.Identity{}
	}
	id := bridge.Identity{JID: cli.Store.ID.ToNonAD().String(), Device: int(cli.Store.ID.Device)}
	if lid := cli.Store.GetLID(); !lid.IsEmpty() {
		id.LID = lid.ToNonAD().String()
	}
	return id
}

func (c *Client) Connect() error { return c.wa().Connect() }

func (c *Client) Disconnect() { c.wa().Disconnect() }

func (c *Client) Close() {
	c.wa().Disconnect()
	_ = c.container.Close()
}

func (c *Client) StartPair(ctx context.Context) (<-chan bridge.QRItem, error) {
	if c.Linked() {
		return nil, errors.New("already linked")
	}
	if c.wa().Store.Deleted {
		c.reset()
	}
	cli := c.wa()
	qr, err := cli.GetQRChannel(ctx)
	if err != nil {
		return nil, err
	}
	if err := cli.Connect(); err != nil {
		return nil, err
	}
	out := make(chan bridge.QRItem, 4)
	go func() {
		defer close(out)
		for it := range qr {
			switch it.Event {
			case whatsmeow.QRChannelEventCode:
				out <- bridge.QRItem{Code: it.Code, Timeout: it.Timeout}
			case whatsmeow.QRChannelSuccess.Event:
				out <- bridge.QRItem{Done: "success"}
				return
			case whatsmeow.QRChannelTimeout.Event:
				cli.Disconnect()
				out <- bridge.QRItem{Done: "timeout"}
				return
			case whatsmeow.QRChannelEventError:
				out <- bridge.QRItem{Done: "error", Err: "pair error"}
				return
			default:
				out <- bridge.QRItem{Done: "error", Err: it.Event}
				return
			}
		}
	}()
	return out, nil
}

func (c *Client) Logout(ctx context.Context) error {
	if err := c.wa().Logout(ctx); err != nil {
		return err
	}
	c.reset()
	return nil
}

func (c *Client) self() (pn, lid types.JID) {
	cli := c.wa()
	if cli.Store.ID != nil {
		pn = cli.Store.ID.ToNonAD()
	}
	if l := cli.Store.GetLID(); !l.IsEmpty() {
		lid = l.ToNonAD()
	}
	return
}

func (c *Client) isSelf(j types.JID) bool {
	pn, lid := c.self()
	j = j.ToNonAD()
	return (!pn.IsEmpty() && j == pn) || (!lid.IsEmpty() && j == lid)
}

func (c *Client) selfChat() bridge.ChatInfo {
	pn, _ := c.self()
	return bridge.ChatInfo{ID: pn.String(), Subject: "Message yourself", Members: 1, OwnerOnly: true, IsSelf: true}
}

func (c *Client) groupInfo(g *types.GroupInfo) bridge.ChatInfo {
	owner := len(g.Participants) > 0
	for _, p := range g.Participants {
		if !(c.isSelf(p.JID) || (!p.LID.IsEmpty() && c.isSelf(p.LID)) || (!p.PhoneNumber.IsEmpty() && c.isSelf(p.PhoneNumber))) {
			owner = false
			break
		}
	}
	n := len(g.Participants)
	if g.ParticipantCount > n {
		n = g.ParticipantCount
	}
	return bridge.ChatInfo{ID: g.JID.String(), Subject: g.Name, Members: n, OwnerOnly: owner && n <= 1}
}

func (c *Client) ListChats(ctx context.Context) ([]bridge.ChatInfo, error) {
	groups, err := c.wa().GetJoinedGroups(ctx)
	if err != nil {
		return nil, err
	}
	out := []bridge.ChatInfo{c.selfChat()}
	for _, g := range groups {
		out = append(out, c.groupInfo(g))
	}
	return out, nil
}

func (c *Client) GetChat(ctx context.Context, chat string) (bridge.ChatInfo, error) {
	if self := c.selfChat(); chat == self.ID {
		return self, nil
	}
	jid, err := types.ParseJID(chat)
	if err != nil {
		return bridge.ChatInfo{}, err
	}
	g, err := c.wa().GetGroupInfo(ctx, jid)
	if err != nil {
		return bridge.ChatInfo{}, err
	}
	return c.groupInfo(g), nil
}

func parseChat(chat string) (types.JID, error) {
	j, err := types.ParseJID(chat)
	if err != nil {
		return j, err
	}
	return j.ToNonAD(), nil
}

func (c *Client) SendText(ctx context.Context, chat, text string, reply *bridge.ReplyRef, mentions []string) (string, error) {
	jid, err := parseChat(chat)
	if err != nil {
		return "", err
	}
	resp, err := c.wa().SendMessage(ctx, jid, textMessage(text, c.quoteContext(reply), mentions))
	return resp.ID, err
}

// textMessage is a plain Conversation, or an ExtendedTextMessage when it
// quotes or mentions.
func textMessage(text string, ci *waE2E.ContextInfo, mentions []string) *waE2E.Message {
	if len(mentions) > 0 {
		if ci == nil {
			ci = &waE2E.ContextInfo{}
		}
		ci.MentionedJID = append([]string{}, mentions...)
	}
	if ci == nil {
		return &waE2E.Message{Conversation: proto.String(text)}
	}
	return &waE2E.Message{ExtendedTextMessage: &waE2E.ExtendedTextMessage{Text: proto.String(text), ContextInfo: ci}}
}

func (c *Client) quoteContext(reply *bridge.ReplyRef) *waE2E.ContextInfo {
	if reply == nil {
		return nil
	}
	participant := reply.Sender
	if participant == "" {
		pn, _ := c.self()
		participant = pn.String()
	}
	return &waE2E.ContextInfo{
		StanzaID:      proto.String(reply.MsgID),
		Participant:   proto.String(participant),
		QuotedMessage: &waE2E.Message{Conversation: proto.String("")},
	}
}

func (c *Client) Edit(ctx context.Context, chat, msgID, text string, mentions []string) error {
	jid, err := parseChat(chat)
	if err != nil {
		return err
	}
	cli := c.wa()
	_, err = cli.SendMessage(ctx, jid, cli.BuildEdit(jid, msgID, textMessage(text, nil, mentions)))
	return err
}

// Members lists a group's participants with display names (contact name,
// else push name). The self chat has only the owner.
func (c *Client) Members(ctx context.Context, chat string) ([]bridge.Member, error) {
	jid, err := parseChat(chat)
	if err != nil {
		return nil, err
	}
	if jid.Server != types.GroupServer {
		pn, _ := c.self()
		return []bridge.Member{{JID: pn.String(), Number: pn.User}}, nil
	}
	g, err := c.wa().GetGroupInfo(ctx, jid)
	if err != nil {
		return nil, err
	}
	out := make([]bridge.Member, 0, len(g.Participants))
	for _, p := range g.Participants {
		name := c.nameOf(ctx, p.JID, "")
		if name == "" && !p.PhoneNumber.IsEmpty() {
			name = c.nameOf(ctx, p.PhoneNumber, "")
		}
		out = append(out, bridge.Member{JID: p.JID.ToNonAD().String(), Number: p.JID.User, Name: name})
	}
	return out, nil
}

// nameOf is the contact name for a user, else its push name, else fallback.
func (c *Client) nameOf(ctx context.Context, j types.JID, fallback string) string {
	if j.IsEmpty() {
		return fallback
	}
	contacts := c.wa().Store.Contacts
	if contacts == nil { // no paired device yet
		return fallback
	}
	ci, err := contacts.GetContact(ctx, j.ToNonAD())
	if err != nil || !ci.Found {
		return fallback
	}
	for _, n := range []string{ci.FullName, ci.FirstName, ci.PushName} {
		if n != "" {
			return n
		}
	}
	return fallback
}

// DownloadToFile streams quoted media into f; f refuses bytes past its cap.
func (c *Client) DownloadToFile(ctx context.Context, m *bridge.MediaIn, f *bridge.CappedFile) error {
	dm, ok := m.Handle.(whatsmeow.DownloadableMessage)
	if !ok {
		return errors.New("not downloadable")
	}
	return c.wa().DownloadToFile(ctx, dm, f)
}

// RequestHistory asks the owner's phone for messages before `before`.
func (c *Client) RequestHistory(ctx context.Context, before bridge.MsgMeta, count int) error {
	jid, err := parseChat(before.Chat)
	if err != nil {
		return err
	}
	cli := c.wa()
	info := &types.MessageInfo{
		MessageSource: types.MessageSource{Chat: jid, IsFromMe: before.FromMe},
		ID:            before.ID,
		Timestamp:     before.At,
	}
	_, err = cli.SendPeerMessage(ctx, cli.BuildHistorySyncRequest(info, count))
	return err
}

// FetchHistory downloads an on-demand history blob and parses every message.
// Media in it is never downloaded (convert reads no media).
func (c *Client) FetchHistory(ctx context.Context, notice any) ([]*bridge.Inbound, error) {
	n, ok := notice.(*waE2E.HistorySyncNotification)
	if !ok {
		return nil, errors.New("not a history notification")
	}
	hs, err := c.wa().DownloadHistorySync(ctx, n, false)
	if err != nil {
		return nil, err
	}
	var out []*bridge.Inbound
	for _, conv := range hs.GetConversations() {
		chat, err := types.ParseJID(conv.GetID())
		if err != nil {
			continue
		}
		for _, hm := range conv.GetMessages() {
			evt, err := c.wa().ParseWebMessage(chat, hm.GetMessage())
			if err != nil {
				continue
			}
			out = append(out, c.convert(evt))
		}
	}
	return out, nil
}

func (c *Client) React(ctx context.Context, chat, sender, msgID, emoji string) error {
	jid, err := parseChat(chat)
	if err != nil {
		return err
	}
	senderJID := types.EmptyJID // our own message
	if sender != "" {
		if senderJID, err = types.ParseJID(sender); err != nil {
			return err
		}
	}
	cli := c.wa()
	_, err = cli.SendMessage(ctx, jid, cli.BuildReaction(jid, senderJID, msgID, emoji))
	return err
}

func (c *Client) SendMedia(ctx context.Context, m bridge.MediaOut) (string, error) {
	jid, err := parseChat(m.Chat)
	if err != nil {
		return "", err
	}
	cli := c.wa()
	var msg *waE2E.Message
	switch m.Kind {
	case "image":
		up, err := cli.Upload(ctx, m.Data, whatsmeow.MediaImage)
		if err != nil {
			return "", err
		}
		img := &waE2E.ImageMessage{
			URL: proto.String(up.URL), DirectPath: proto.String(up.DirectPath), MediaKey: up.MediaKey,
			FileEncSHA256: up.FileEncSHA256, FileSHA256: up.FileSHA256, FileLength: proto.Uint64(up.FileLength),
			Mimetype: proto.String(m.Mime),
		}
		if m.Caption != "" {
			img.Caption = proto.String(m.Caption)
		}
		msg = &waE2E.Message{ImageMessage: img}
	case "document":
		up, err := cli.Upload(ctx, m.Data, whatsmeow.MediaDocument)
		if err != nil {
			return "", err
		}
		doc := &waE2E.DocumentMessage{
			URL: proto.String(up.URL), DirectPath: proto.String(up.DirectPath), MediaKey: up.MediaKey,
			FileEncSHA256: up.FileEncSHA256, FileSHA256: up.FileSHA256, FileLength: proto.Uint64(up.FileLength),
			Mimetype: proto.String(m.Mime), FileName: proto.String(m.FileName), Title: proto.String(m.FileName),
		}
		if m.Caption != "" {
			doc.Caption = proto.String(m.Caption)
			msg = &waE2E.Message{DocumentWithCaptionMessage: &waE2E.FutureProofMessage{Message: &waE2E.Message{DocumentMessage: doc}}}
		} else {
			msg = &waE2E.Message{DocumentMessage: doc}
		}
	default:
		return "", errors.New("unsupported kind")
	}
	resp, err := cli.SendMessage(ctx, jid, msg)
	return resp.ID, err
}

func (c *Client) SetPresence(ctx context.Context, available bool) error {
	p := types.PresenceUnavailable
	if available {
		p = types.PresenceAvailable
	}
	return c.wa().SendPresence(ctx, p)
}

func (c *Client) ChatPresence(ctx context.Context, chat string, composing bool) error {
	jid, err := parseChat(chat)
	if err != nil {
		return err
	}
	s := types.ChatPresencePaused
	if composing {
		s = types.ChatPresenceComposing
	}
	return c.wa().SendChatPresence(ctx, jid, s, types.ChatPresenceMediaText)
}

func (c *Client) Download(ctx context.Context, m *bridge.MediaIn) ([]byte, error) {
	dm, ok := m.Handle.(whatsmeow.DownloadableMessage)
	if !ok {
		return nil, errors.New("not downloadable")
	}
	return c.wa().Download(ctx, dm)
}

// ---- events ----------------------------------------------------------------

func (c *Client) onWAEvent(evt any) {
	switch v := evt.(type) {
	case *events.Connected:
		c.emit(bridge.Event{Connected: true})
	case *events.Disconnected:
		c.emit(bridge.Event{Offline: "disconnected"})
	case *events.StreamReplaced:
		c.emit(bridge.Event{Offline: "replaced"})
	case *events.TemporaryBan:
		c.emit(bridge.Event{Offline: "temp_banned"})
	case *events.ClientOutdated:
		c.emit(bridge.Event{Offline: "client_outdated"})
	case *events.ConnectFailure:
		c.emit(bridge.Event{Offline: "connect_failure"})
	case *events.LoggedOut:
		c.emit(bridge.Event{LoggedOut: logoutReason(v)})
		go c.reset()
	case *events.OfflineSyncPreview:
		c.emit(bridge.Event{SyncStart: true})
	case *events.OfflineSyncCompleted:
		c.emit(bridge.Event{SyncDone: true})
	case *events.GroupInfo:
		c.emit(bridge.Event{ChatChanged: v.JID.ToNonAD().String()})
	case *events.Message:
		// History sync notifications are not downloaded (ManualHistorySyncDownload);
		// only on-demand answers go to the bridge, which processes them only
		// while it has an open request.
		if n := v.Message.GetProtocolMessage().GetHistorySyncNotification(); n != nil {
			if n.GetSyncType() == waE2E.HistorySyncType_ON_DEMAND && c.isSelf(v.Info.Sender) {
				c.emit(bridge.Event{HistoryNotice: n})
			}
			return
		}
		if pm := v.Message.GetProtocolMessage(); pm != nil && pm.GetType() == waE2E.ProtocolMessage_REVOKE {
			if id := pm.GetKey().GetID(); id != "" {
				c.emit(bridge.Event{Revoked: &bridge.Revoked{Chat: v.Info.Chat.ToNonAD().String(), MsgID: id}})
			}
			return
		}
		c.emit(bridge.Event{Message: c.convert(v)})
	case *events.DeleteForMe:
		if v.MessageID != "" {
			c.emit(bridge.Event{Revoked: &bridge.Revoked{Chat: v.ChatJID.ToNonAD().String(), MsgID: v.MessageID}})
		}
	}
}

func logoutReason(v *events.LoggedOut) string {
	switch v.Reason {
	case events.ConnectFailureLoggedOut:
		return "phone"
	case events.ConnectFailureUnknownLogout: // "BANNED" in WhatsApp Web
		return "banned"
	}
	if !v.OnConnect {
		return "phone" // device_removed stream error: unlinked from the phone
	}
	return "other"
}

// convert turns a whatsmeow message into the bridge's neutral form. It reads
// no media and never logs.
func (c *Client) convert(v *events.Message) *bridge.Inbound {
	info := v.Info
	chat := info.Chat.ToNonAD()
	// "Message yourself" may arrive addressed by LID: report it by phone JID.
	if pn, lid := c.self(); !lid.IsEmpty() && chat == lid && !pn.IsEmpty() {
		chat = pn
	}
	in := &bridge.Inbound{
		Chat:         chat.String(),
		MsgID:        info.ID,
		SenderUser:   info.Sender.ToNonAD().String(),
		SenderDevice: int(info.Sender.Device),
		SenderName:   c.nameOf(context.Background(), info.Sender, info.PushName),
		IsFromMe:     info.IsFromMe,
		Timestamp:    info.Timestamp,
		IsStatus:     info.Chat == types.StatusBroadcastJID,
		IsBroadcast:  info.Chat.Server == types.BroadcastServer,
	}
	if !info.SenderAlt.IsEmpty() {
		in.SenderAlt = info.SenderAlt.ToNonAD().String()
	}
	m := v.Message
	if m == nil {
		in.IsProtocol = true
		return in
	}
	if v.IsEdit || m.GetProtocolMessage() != nil {
		in.IsProtocol = true
		return in
	}
	var ctxInfo *waE2E.ContextInfo
	switch {
	case m.GetConversation() != "":
		in.Text = m.GetConversation()
	case m.GetExtendedTextMessage() != nil:
		in.Text = m.GetExtendedTextMessage().GetText()
		ctxInfo = m.GetExtendedTextMessage().GetContextInfo()
	case m.GetReactionMessage() != nil:
		r := m.GetReactionMessage()
		in.Reaction = &bridge.Reaction{Emoji: r.GetText(), TargetMsg: r.GetKey().GetID()}
	case m.GetImageMessage() != nil:
		x := m.GetImageMessage()
		in.Text, ctxInfo = x.GetCaption(), x.GetContextInfo()
		in.Media = &bridge.MediaIn{Kind: "image", Mime: x.GetMimetype(), Size: int64(x.GetFileLength()), Handle: x}
	case m.GetDocumentMessage() != nil:
		x := m.GetDocumentMessage()
		in.Text, ctxInfo = x.GetCaption(), x.GetContextInfo()
		in.Media = &bridge.MediaIn{Kind: "document", Mime: x.GetMimetype(), FileName: x.GetFileName(), Size: int64(x.GetFileLength()), Handle: x}
	case m.GetAudioMessage() != nil:
		kind := "audio"
		if m.GetAudioMessage().GetPTT() {
			kind = "voice"
		}
		in.Media = &bridge.MediaIn{Kind: kind} // no handle: never downloadable
	case m.GetVideoMessage() != nil || m.GetPtvMessage() != nil:
		in.Text = m.GetVideoMessage().GetCaption()
		in.Media = &bridge.MediaIn{Kind: "video"}
	case m.GetStickerMessage() != nil:
		in.Media = &bridge.MediaIn{Kind: "sticker"}
	}
	if ctxInfo != nil && ctxInfo.GetStanzaID() != "" {
		qm := ctxInfo.GetQuotedMessage()
		q := &bridge.Quoted{MsgID: ctxInfo.GetStanzaID(), Text: textOf(qm), Links: previewLinks(qm), Media: mediaOf(qm)}
		if r, err := types.ParseJID(ctxInfo.GetRemoteJID()); err == nil && !r.IsEmpty() {
			q.RemoteJID = r.ToNonAD().String()
			if pn, lid := c.self(); !lid.IsEmpty() && r.ToNonAD() == lid && !pn.IsEmpty() {
				q.RemoteJID = pn.String()
			}
		}
		if p, err := types.ParseJID(ctxInfo.GetParticipant()); err == nil && !p.IsEmpty() {
			q.Participant = p.ToNonAD().String()
			q.SenderName = c.nameOf(context.Background(), p, "")
			if c.isSelf(p) {
				pn, _ := c.self()
				q.Participant = pn.String()
			}
		}
		in.Quoted = q
	}
	return in
}

// previewLinks are the URLs of a quoted message's link preview.
func previewLinks(m *waE2E.Message) []string {
	x := m.GetExtendedTextMessage()
	if x == nil {
		return nil
	}
	var out []string
	for _, u := range []string{x.GetMatchedText()} {
		if u != "" {
			out = append(out, u)
		}
	}
	return out
}

// mediaOf describes a quoted message's media. Only images and documents get a
// download handle; the bridge decides whether to use it.
func mediaOf(m *waE2E.Message) *bridge.MediaIn {
	switch {
	case m == nil:
		return nil
	case m.GetImageMessage() != nil:
		x := m.GetImageMessage()
		return &bridge.MediaIn{Kind: "image", Mime: x.GetMimetype(), Size: int64(x.GetFileLength()), Handle: x}
	case m.GetDocumentMessage() != nil:
		x := m.GetDocumentMessage()
		return &bridge.MediaIn{Kind: "document", Mime: x.GetMimetype(), FileName: x.GetFileName(), Size: int64(x.GetFileLength()), Handle: x}
	case m.GetAudioMessage() != nil:
		if m.GetAudioMessage().GetPTT() {
			return &bridge.MediaIn{Kind: "voice"}
		}
		return &bridge.MediaIn{Kind: "audio"}
	case m.GetVideoMessage() != nil || m.GetPtvMessage() != nil:
		return &bridge.MediaIn{Kind: "video"}
	case m.GetStickerMessage() != nil:
		return &bridge.MediaIn{Kind: "sticker"}
	}
	return nil
}

func textOf(m *waE2E.Message) string {
	switch {
	case m == nil:
		return ""
	case m.GetConversation() != "":
		return m.GetConversation()
	case m.GetExtendedTextMessage() != nil:
		return m.GetExtendedTextMessage().GetText()
	case m.GetImageMessage() != nil:
		return m.GetImageMessage().GetCaption()
	case m.GetDocumentMessage() != nil:
		return m.GetDocumentMessage().GetCaption()
	}
	return ""
}

// logAdapter forwards whatsmeow warnings and errors to stderr, truncated.
// Info and debug are dropped: they can carry node dumps.
type logAdapter struct {
	log    *bridge.Logger
	module string
}

func (l *logAdapter) out(level, msg string, args ...any) {
	s := fmt.Sprintf(msg, args...)
	if len(s) > 300 {
		s = s[:300] + "..."
	}
	l.log.Printf("whatsmeow %s %s: %s", level, l.module, s)
}

func (l *logAdapter) Warnf(msg string, args ...any)  { l.out("WARN", msg, args...) }
func (l *logAdapter) Errorf(msg string, args ...any) { l.out("ERROR", msg, args...) }
func (l *logAdapter) Infof(string, ...any)           {}
func (l *logAdapter) Debugf(string, ...any)          {}
func (l *logAdapter) Sub(module string) waLog.Logger {
	m := module
	if l.module != "" {
		m = l.module + "/" + module
	}
	return &logAdapter{log: l.log, module: m}
}
