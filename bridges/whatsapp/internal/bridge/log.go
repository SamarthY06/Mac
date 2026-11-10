package bridge

import (
	"fmt"
	"io"
	"strings"
	"sync"
	"time"
)

// Logger writes stderr lines. Callers pass ids, kinds, counts and lengths
// only, never message text, captions, subjects or file names.
type Logger struct {
	mu sync.Mutex
	w  io.Writer
}

func NewLogger(w io.Writer) *Logger { return &Logger{w: w} }

func (l *Logger) Printf(format string, args ...any) {
	if l == nil || l.w == nil {
		return
	}
	line := fmt.Sprintf(format, args...)
	line = strings.ReplaceAll(line, "\n", " ")
	l.mu.Lock()
	defer l.mu.Unlock()
	fmt.Fprintf(l.w, "%s wa-bridge %s\n", time.Now().UTC().Format("2006-01-02T15:04:05.000Z"), line)
}
