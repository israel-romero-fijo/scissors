# Security boundaries

Public tools accept operation names and validated JSON, never Python snippets. Dispatch
uses registered callables. Paths resolve symlinks and must remain in the workspace;
state.json, .workspace.lock, plugins and .runtime are reserved from artifact APIs.
No outside-workspace export bypass is provided. Plugin packages reject symlinks and excess size.

FileLock enforces one owner; RLock serializes calls. Important writes use fsync and atomic
replacement. Atomic batches cover scene data/staged plugin geometry, rejecting artifacts
and plugin lifecycle effects. Multi-file exports are individually atomic; unreferenced
artifacts may remain after filesystem failure, but the scene checkpoint remains coherent.

Settings cap geometry counts, estimated memory, image pixels, files and worker time.
Native operations are bounded but not all preemptively cancellable. POSIX plugin workers
use RLIMIT_AS/RLIMIT_FSIZE; Windows uses timeout/output limits without a hard memory
ceiling. This is not a hardened hostile-code executor.

Plugins use an application plugin store outside workspaces (default ~/.scissors/plugins).
Every workspace session discovers valid packages there and enables them unless explicitly
disabled in that workspace. Installation affects every workspace using the store. Writing
to this shared store is therefore a code-execution decision. A shared file lock serializes
package changes and worker access. Workspace artifact APIs cannot reach the store.
Public transports cannot install
or uninstall packages, and artifact APIs keep plugins reserved, so a remote client can only
enable, disable or reload packages already present. Package code retains OS/network access.

Worker environments omit application secrets. Server error logs contain operation names
and exception types, rather than payloads. Plugin errors return truncated messages for
diagnosis; plugin authors must avoid secrets in exception messages/test output.

HTTP defaults to 127.0.0.1, verifies loopback host and origin, rejects cross-origin requests
and bounds bodies including chunked input. MCP accepts loopback headers for `127.0.0.1`,
`localhost` and `[::1]` with an explicit port or with the default port omitted; an Origin
header must still match the request's scheme and Host. Bearer authentication, when set, protects REST,
MCP and OpenAPI. Other bind addresses require SCISSORS_TOKEN; provide TLS via a proxy.
Scissors does not provide an OAuth server or per-user role-based access control. Read/write metadata describes effects;
authenticated clients are model-editing operators.

Close workspaces cleanly; never remove a live lock or describe subprocesses as secure isolation.


MCP HTTP responses delegate body framing to the ASGI HTTP server. Fixed
Content-Length headers are omitted on `/mcp` responses, including JSON replies and
errors. Response bytes, status, and streaming/disconnect behavior are retained.
REST response lengths are unchanged. Request body size checks still apply.
