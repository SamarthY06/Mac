// Renders the Mac app icon into an .iconset directory (all .icns sizes, 1x and 2x).
//   swift scripts/make_icon.swift <out.iconset>   then   iconutil -c icns <out.iconset>
//
// Design (docs/DESIGN.md): the HIG macOS icon grid — a 1024 pt canvas with an 824 pt
// rounded-square body (continuous corners, ~22.4 % radius) and a soft drop shadow in the
// margin — carrying one simple mark: a glowing orb with a five-bar waveform. Filled,
// overlapping shapes only; no text, no fine lines, so it reads at 16 px. Everything is
// drawn with CoreGraphics; no third-party assets.
import AppKit

let out = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "Mac.iconset"
try? FileManager.default.createDirectory(atPath: out, withIntermediateDirectories: true)

/// Continuous-curvature ("squircle") rounded rect, close to the system icon mask.
func squircle(_ r: CGRect, radius: CGFloat) -> CGPath {
    let p = CGMutablePath()
    let k: CGFloat = 1.2  // control-point stretch → smoother than a circular corner
    let c = min(radius * k, min(r.width, r.height) / 2)
    let (x0, y0, x1, y1) = (r.minX, r.minY, r.maxX, r.maxY)
    p.move(to: CGPoint(x: x0 + c, y: y0))
    p.addLine(to: CGPoint(x: x1 - c, y: y0))
    p.addCurve(to: CGPoint(x: x1, y: y0 + c), control1: CGPoint(x: x1 - c * 0.45, y: y0), control2: CGPoint(x: x1, y: y0 + c * 0.45))
    p.addLine(to: CGPoint(x: x1, y: y1 - c))
    p.addCurve(to: CGPoint(x: x1 - c, y: y1), control1: CGPoint(x: x1, y: y1 - c * 0.45), control2: CGPoint(x: x1 - c * 0.45, y: y1))
    p.addLine(to: CGPoint(x: x0 + c, y: y1))
    p.addCurve(to: CGPoint(x: x0, y: y1 - c), control1: CGPoint(x: x0 + c * 0.45, y: y1), control2: CGPoint(x: x0, y: y1 - c * 0.45))
    p.addLine(to: CGPoint(x: x0, y: y0 + c))
    p.addCurve(to: CGPoint(x: x0 + c, y: y0), control1: CGPoint(x: x0, y: y0 + c * 0.45), control2: CGPoint(x: x0 + c * 0.45, y: y0))
    p.closeSubpath()
    return p
}

func rgb(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ a: CGFloat = 1) -> CGColor {
    CGColor(colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!, components: [r, g, b, a])!
}

func render(_ px: Int) -> Data {
    let s = CGFloat(px)
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: px, pixelsHigh: px, bitsPerSample: 8,
                               samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                               colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    let ctx = NSGraphicsContext.current!.cgContext
    ctx.setAllowsAntialiasing(true)
    ctx.interpolationQuality = .high
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!

    // Body: 824/1024 of the canvas, centred (HIG macOS icon grid).
    let inset = s * (100.0 / 1024.0)
    let body = CGRect(x: inset, y: inset, width: s - 2 * inset, height: s - 2 * inset)
    let path = squircle(body, radius: body.width * 0.2237)

    // Drop shadow in the margin (small sizes get a lighter one so the edge stays crisp).
    ctx.saveGState()
    ctx.setShadow(offset: CGSize(width: 0, height: -s * 0.012), blur: s * 0.028,
                  color: CGColor(gray: 0, alpha: px >= 64 ? 0.32 : 0.18))
    ctx.addPath(path); ctx.setFillColor(rgb(0.10, 0.12, 0.30)); ctx.fillPath()
    ctx.restoreGState()

    ctx.saveGState()
    ctx.addPath(path); ctx.clip()

    // Background: calm indigo → blue → aqua diagonal gradient.
    let bg = CGGradient(colorsSpace: cs, colors: [
        rgb(0.13, 0.11, 0.42),
        rgb(0.16, 0.36, 0.86),
        rgb(0.33, 0.76, 0.93),
    ] as CFArray, locations: [0, 0.55, 1])!
    ctx.drawLinearGradient(bg, start: CGPoint(x: body.minX, y: body.maxY), end: CGPoint(x: body.maxX, y: body.minY), options: [])

    // Soft top light + bottom depth (subtle; the system adds its own on macOS 26).
    let light = CGGradient(colorsSpace: cs, colors: [rgb(1, 1, 1, 0.16), rgb(1, 1, 1, 0)] as CFArray, locations: [0, 1])!
    ctx.drawLinearGradient(light, start: CGPoint(x: body.midX, y: body.maxY), end: CGPoint(x: body.midX, y: body.midY), options: [])
    let depth = CGGradient(colorsSpace: cs, colors: [rgb(0, 0, 0.1, 0), rgb(0, 0, 0.1, 0.18)] as CFArray, locations: [0, 1])!
    ctx.drawLinearGradient(depth, start: CGPoint(x: body.midX, y: body.midY), end: CGPoint(x: body.midX, y: body.minY), options: [])

    // Orb: a soft radial glow, then a translucent disc (filled, overlapping shapes).
    let c = CGPoint(x: s / 2, y: s / 2)
    let glow = CGGradient(colorsSpace: cs, colors: [
        rgb(1, 1, 1, 0.40), rgb(0.75, 0.92, 1, 0.14), rgb(0.75, 0.92, 1, 0),
    ] as CFArray, locations: [0, 0.55, 1])!
    ctx.drawRadialGradient(glow, startCenter: c, startRadius: 0, endCenter: c, endRadius: body.width * 0.46, options: [])
    let discR = body.width * 0.30
    ctx.setFillColor(rgb(1, 1, 1, 0.14))
    ctx.fillEllipse(in: CGRect(x: c.x - discR, y: c.y - discR, width: discR * 2, height: discR * 2))

    // Waveform: five rounded bars, symmetric, white.
    let heights: [CGFloat] = [0.22, 0.40, 0.58, 0.40, 0.22]
    let barW = body.width * 0.072, gap = body.width * 0.052
    let total = CGFloat(heights.count) * barW + CGFloat(heights.count - 1) * gap
    var x = c.x - total / 2
    ctx.setShadow(offset: .zero, blur: s * 0.015, color: rgb(0.1, 0.2, 0.6, 0.35))
    ctx.setFillColor(rgb(1, 1, 1, 0.97))
    for h in heights {
        let bh = body.width * h
        let r = CGRect(x: x, y: c.y - bh / 2, width: barW, height: bh)
        ctx.addPath(CGPath(roundedRect: r, cornerWidth: barW / 2, cornerHeight: barW / 2, transform: nil))
        ctx.fillPath()
        x += barW + gap
    }
    ctx.restoreGState()
    NSGraphicsContext.restoreGraphicsState()
    return rep.representation(using: .png, properties: [:])!
}

for base in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let name = scale == 1 ? "icon_\(base)x\(base).png" : "icon_\(base)x\(base)@2x.png"
        try! render(base * scale).write(to: URL(fileURLWithPath: out).appendingPathComponent(name))
    }
}
print("wrote \(out)")
