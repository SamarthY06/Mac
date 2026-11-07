// NDJSON line splitting for control.sock (ARCHITECTURE.md §11).
import Foundation

public struct LineDecoder: Sendable {
    public let maxLine: Int
    private var buf = Data()

    public init(maxLine: Int = 4 * 1024 * 1024) { self.maxLine = maxLine }

    /// Returns complete lines (without the trailing "\n", "\r" stripped). Lines longer than
    /// `maxLine` are dropped; empty lines are skipped.
    public mutating func feed(_ chunk: Data) -> [Data] {
        buf.append(chunk)
        var lines: [Data] = []
        var start = buf.startIndex
        while let nl = buf[start...].firstIndex(of: 0x0A) {
            var line = buf.subdata(in: start..<nl)
            if line.last == 0x0D { line.removeLast() }
            if !line.isEmpty && line.count <= maxLine { lines.append(line) }
            start = buf.index(after: nl)
        }
        buf = buf.subdata(in: start..<buf.endIndex)
        if buf.count > maxLine { buf = Data() }  // runaway line without newline
        return lines
    }
}
