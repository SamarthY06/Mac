// Command wa-bridge is the WhatsApp sidecar of chatd. It speaks the stdio
// protocol in docs/whatsapp/bridge-protocol.md.
package main

import (
	"os"
	"syscall"

	"github.com/SamarthY06/OpenMac/bridges/whatsapp/internal/bridge"
	"github.com/SamarthY06/OpenMac/bridges/whatsapp/internal/wa"
)

func main() {
	syscall.Umask(0o077) // store, WAL, lock and media files: owner-only
	os.Exit(bridge.Run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr, bridge.Options{Open: wa.Open}))
}
