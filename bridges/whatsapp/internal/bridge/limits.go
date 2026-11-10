package bridge

import (
	"encoding/json"
	"os"
	"path/filepath"
	"time"
)

// window is a rolling one-minute counter.
type window struct {
	limit int
	times []time.Time
}

func (w *window) prune(now time.Time) {
	cut := now.Add(-time.Minute)
	i := 0
	for i < len(w.times) && !w.times[i].After(cut) {
		i++
	}
	w.times = w.times[i:]
}

func (w *window) full(now time.Time) bool {
	w.prune(now)
	return len(w.times) >= w.limit
}

func (w *window) add(now time.Time) { w.times = append(w.times, now) }

func (w *window) count(now time.Time) int {
	w.prune(now)
	return len(w.times)
}

// sentEntry records one message this bridge sent. No text is kept.
type sentEntry struct {
	ID   string    `json:"id"`
	Chat string    `json:"chat"`
	At   time.Time `json:"at"`
}

// sentRing holds the last sentRingSize ids this bridge sent (echo rule 4 and
// edit ownership). It is persisted beside the store, so edits and echo
// detection survive a bridge restart.
type sentRing struct {
	path    string
	entries []sentEntry
	byID    map[string]int // id -> count of occurrences in entries
}

const sentRingSize = 2000

func loadSentRing(path string) *sentRing {
	r := &sentRing{path: path, byID: map[string]int{}}
	if path == "" {
		return r
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return r
	}
	var es []sentEntry
	if json.Unmarshal(b, &es) != nil {
		return r
	}
	for _, e := range es {
		r.push(e)
	}
	return r
}

func (r *sentRing) push(e sentEntry) {
	r.entries = append(r.entries, e)
	r.byID[e.ID]++
	for len(r.entries) > sentRingSize {
		old := r.entries[0]
		r.entries = r.entries[1:]
		if r.byID[old.ID]--; r.byID[old.ID] <= 0 {
			delete(r.byID, old.ID)
		}
	}
}

func (r *sentRing) add(e sentEntry) {
	r.push(e)
	r.save()
}

func (r *sentRing) has(id string) bool { return r.byID[id] > 0 }

func (r *sentRing) get(id string) (sentEntry, bool) {
	if !r.has(id) {
		return sentEntry{}, false
	}
	for i := len(r.entries) - 1; i >= 0; i-- {
		if r.entries[i].ID == id {
			return r.entries[i], true
		}
	}
	return sentEntry{}, false
}

func (r *sentRing) save() {
	if r.path == "" {
		return
	}
	b, err := json.Marshal(r.entries)
	if err != nil {
		return
	}
	tmp := r.path + ".tmp"
	if os.WriteFile(tmp, b, 0o600) == nil {
		_ = os.Rename(tmp, filepath.Clean(r.path))
	}
}

// senderCache maps recent message ids in allowlisted chats to their author,
// time and from-me flag, so react/reply_to with from_me=false can address the
// right participant and `history` can name the message to start before.
// Only ids, JIDs and times are kept, never content.
type senderCache struct {
	order []string
	m     map[string]MsgMeta
	who   map[string]string
}

const senderCacheSize = 2000

func newSenderCache() *senderCache {
	return &senderCache{m: map[string]MsgMeta{}, who: map[string]string{}}
}

func (c *senderCache) put(meta MsgMeta, sender string) {
	k := meta.Chat + "/" + meta.ID
	if _, ok := c.m[k]; !ok {
		c.order = append(c.order, k)
	}
	c.m[k], c.who[k] = meta, sender
	for len(c.order) > senderCacheSize {
		delete(c.m, c.order[0])
		delete(c.who, c.order[0])
		c.order = c.order[1:]
	}
}

func (c *senderCache) get(chat, id string) (string, bool) {
	s, ok := c.who[chat+"/"+id]
	return s, ok
}

func (c *senderCache) meta(chat, id string) (MsgMeta, bool) {
	m, ok := c.m[chat+"/"+id]
	return m, ok
}
