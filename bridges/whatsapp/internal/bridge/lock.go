package bridge

import (
	"errors"
	"os"
	"path/filepath"
	"syscall"
)

var errLocked = errors.New("store locked by another bridge")

// lockStore takes flock(LOCK_EX|LOCK_NB) on <store>.lock. It creates the
// store's directory (0700) and the lock file (0600) but never touches the
// store itself. The returned file must stay open for the process lifetime.
func lockStore(store string) (*os.File, error) {
	dir := filepath.Dir(store)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(store+".lock", os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, err
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		f.Close()
		if errors.Is(err, syscall.EWOULDBLOCK) || errors.Is(err, syscall.EAGAIN) {
			return nil, errLocked
		}
		return nil, err
	}
	return f, nil
}
