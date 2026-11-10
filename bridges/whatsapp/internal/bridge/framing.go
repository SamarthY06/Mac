package bridge

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"sync"
)

// MaxLine is the largest accepted frame, newline excluded.
const MaxLine = 1 << 20

var errTooLong = errors.New("line over 1 MiB")

// lineReader yields one line at a time. A line over MaxLine is consumed up to
// its newline and reported as errTooLong, so the next read starts cleanly.
type lineReader struct{ r *bufio.Reader }

func newLineReader(r io.Reader) *lineReader { return &lineReader{bufio.NewReaderSize(r, 64<<10)} }

// next returns (line, nil), (nil, errTooLong) or (nil, io.EOF).
func (l *lineReader) next() ([]byte, error) {
	var buf []byte
	tooLong := false
	for {
		chunk, err := l.r.ReadSlice('\n')
		if !tooLong {
			if len(buf)+len(chunk) > MaxLine+1 { // +1 for the newline
				tooLong = true
				buf = nil
			} else {
				buf = append(buf, chunk...)
			}
		}
		switch {
		case err == nil:
			if tooLong {
				return nil, errTooLong
			}
			return bytes.TrimRight(buf, "\r\n"), nil
		case errors.Is(err, bufio.ErrBufferFull):
			continue
		case errors.Is(err, io.EOF):
			if tooLong {
				return nil, errTooLong
			}
			if len(buf) > 0 { // final unterminated line still counts
				return buf, nil
			}
			return nil, io.EOF
		default:
			return nil, err
		}
	}
}

// writer serialises frames to stdout, one Write per frame.
type writer struct {
	mu sync.Mutex
	w  io.Writer
}

func (w *writer) send(v any) {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		b.Reset()
		b.WriteString(`{"id":null,"ok":false,"code":"internal","error":"encode failed"}` + "\n")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	_, _ = w.w.Write(b.Bytes()) // os.Stdout is unbuffered: written means flushed
}

type okResp struct {
	ID   int64 `json:"id"`
	OK   bool  `json:"ok"`
	Data any   `json:"data"`
}

type errResp struct {
	ID    *int64 `json:"id"`
	OK    bool   `json:"ok"`
	Code  string `json:"code"`
	Error string `json:"error"`
}

func (w *writer) ok(id int64, data any) {
	if data == nil {
		data = struct{}{}
	}
	w.send(okResp{ID: id, OK: true, Data: data})
}

func (w *writer) fail(id *int64, code, msg string) {
	w.send(errResp{ID: id, OK: false, Code: code, Error: msg})
}

// event writes {"ev": name, ...fields}.
func (w *writer) event(name string, fields map[string]any) {
	m := make(map[string]any, len(fields)+1)
	for k, v := range fields {
		m[k] = v
	}
	m["ev"] = name
	w.send(m)
}
