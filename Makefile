.PHONY: app install test daemon-test app-test e2e clean run

APP := build/Mac.app

app:            ## build + sign build/Mac.app
	scripts/bundle.sh

install:        ## full install (deps, models, app → ~/Applications)
	./install.sh

test: app-test daemon-test

app-test:       ## Swift unit tests (audio framing, NDJSON)
	cd app && swift test

daemon-test:    ## Python tests
	cd daemon && uv run pytest -q -m "not live and not models and not perf"

e2e: app        ## app ⇄ fake daemon over real sockets (plays a quiet tone)
	scripts/e2e.sh

run: app        ## run the freshly built app
	open $(APP)

clean:
	rm -rf build app/.build
