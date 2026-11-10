package bridge

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// ProtocolVersion is the stdio protocol version (docs/whatsapp/bridge-protocol.md).
const ProtocolVersion = "2"

// BridgeVersion is reported in hello and ready.
const BridgeVersion = "0.2.0"

// Config is the parsed command line. Every field is fixed for the process
// lifetime; only the allow set is mutable, and only through set_allow (Bridge).
type Config struct {
	Store          string
	MediaDir       string
	Allow          []string
	Context        []string // allowed chats whose non-trigger messages become context events
	Deny           []string
	TrustedAll     bool // --trusted-devices all (default): every device of the owner's account
	TrustedDevices []int
	MaxPerMin      int
	Budget         int // -1 when --budget was not given
	DryRun         bool
	Test           bool
}

// HasBudget reports whether --budget was given.
func (c Config) HasBudget() bool { return c.Budget >= 0 }

// ParseFlags parses and validates argv (without the program name).
// Any error means exit 2.
func ParseFlags(args []string, errOut io.Writer) (Config, error) {
	fs := flag.NewFlagSet("wa-bridge", flag.ContinueOnError)
	fs.SetOutput(errOut)
	var c Config
	var allow, context, deny, trusted, budget string
	fs.StringVar(&c.Store, "store", "", "whatsmeow SQLite store path")
	fs.StringVar(&c.MediaDir, "media-dir", "", "directory for inbound images and documents")
	fs.StringVar(&allow, "allow", "", "comma-separated allowlisted chat JIDs")
	fs.StringVar(&context, "context", "", "comma-separated allowed chats whose other messages are context (subset of --allow)")
	fs.StringVar(&deny, "deny", "", "comma-separated never-listed chat JIDs (immutable)")
	fs.StringVar(&trusted, "trusted-devices", "all", "all, or comma-separated owner device ids")
	fs.IntVar(&c.MaxPerMin, "max-per-min", 20, "new messages per rolling minute")
	fs.StringVar(&budget, "budget", "", "total outbound actions for the process lifetime (immutable)")
	fs.BoolVar(&c.DryRun, "dry-run", false, "perform no outbound network write")
	fs.BoolVar(&c.Test, "test", false, "test mode: requires --budget; refuses set_allow, pair, logout")
	if err := fs.Parse(args); err != nil {
		return c, err
	}
	if fs.NArg() > 0 {
		return c, fmt.Errorf("unexpected arguments: %d", fs.NArg())
	}
	if c.Store == "" {
		return c, errors.New("--store is required")
	}
	if c.MediaDir == "" {
		return c, errors.New("--media-dir is required")
	}
	var err error
	if c.Deny, err = parseJIDList(deny); err != nil {
		return c, fmt.Errorf("--deny: %w", err)
	}
	if c.Allow, err = parseJIDList(allow); err != nil {
		return c, fmt.Errorf("--allow: %w", err)
	}
	for _, a := range c.Allow {
		for _, d := range c.Deny {
			if a == d {
				return c, errors.New("--allow contains a JID that is on --deny")
			}
		}
	}
	if c.Context, err = parseJIDList(context); err != nil {
		return c, fmt.Errorf("--context: %w", err)
	}
	allowed := set(c.Allow)
	for _, j := range c.Context {
		if !allowed[j] {
			return c, errors.New("--context holds a JID that is not on --allow")
		}
	}
	if strings.TrimSpace(trusted) == "all" {
		c.TrustedAll = true
	} else if c.TrustedDevices, err = parseInts(trusted); err != nil || len(c.TrustedDevices) == 0 {
		return c, errors.New("--trusted-devices must be all or a non-empty list of device ids")
	}
	if c.MaxPerMin < 1 {
		return c, errors.New("--max-per-min must be at least 1")
	}
	c.Budget = -1
	if budget != "" {
		n, err := strconv.Atoi(budget)
		if err != nil || n < 0 {
			return c, errors.New("--budget must be a non-negative integer")
		}
		c.Budget = n
	}
	if c.Test && !c.HasBudget() {
		return c, errors.New("--test requires --budget")
	}
	return c, nil
}

func parseJIDList(s string) ([]string, error) {
	out := []string{}
	if strings.TrimSpace(s) == "" {
		return out, nil
	}
	seen := map[string]bool{}
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if !validJID(p) {
			return nil, fmt.Errorf("not a chat JID: %q", p)
		}
		if !seen[p] {
			seen[p] = true
			out = append(out, p)
		}
	}
	return out, nil
}

// validJID accepts user@server with no device part (chat JIDs only).
func validJID(s string) bool {
	at := strings.IndexByte(s, '@')
	if at <= 0 || at == len(s)-1 || strings.Count(s, "@") != 1 {
		return false
	}
	return !strings.ContainsAny(s, " \t\n:,")
}

func parseInts(s string) ([]int, error) {
	var out []int
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil || n < 0 || n > 65535 {
			return nil, fmt.Errorf("bad device id %q", p)
		}
		out = append(out, n)
	}
	return out, nil
}
