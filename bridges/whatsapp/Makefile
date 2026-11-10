# wa-bridge: pure Go (no cgo), see docs/whatsapp/bridge-protocol.md
GO ?= go
BIN := bin/wa-bridge

.PHONY: build test vet clean install

build:
	CGO_ENABLED=0 $(GO) build -trimpath -ldflags="-s -w" -o $(BIN) .

test:
	CGO_ENABLED=0 $(GO) test ./...

vet:
	CGO_ENABLED=0 $(GO) vet ./...

# Installs to the path chatd launches (ARCHITECTURE W5.6).
install: build
	mkdir -p $(HOME)/.mac-agent/bin
	install -m 0755 $(BIN) $(HOME)/.mac-agent/bin/wa-bridge

clean:
	rm -rf bin
