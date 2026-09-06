import AppKit
import Foundation

// A code-native vector mark, rendered at the exact AppIcon sizes.
let directory = URL(fileURLWithPath: CommandLine.arguments[1])
let variants: [(Int, Int)] = [(16,1),(16,2),(32,1),(32,2),(128,1),(128,2),(256,1),(256,2),(512,1),(512,2)]
var entries: [[String: String]] = []
for (size, scale) in variants {
    let pixels = size * scale
    let image = NSImage(size: NSSize(width: pixels, height: pixels))
    image.lockFocus()
    let context = NSGraphicsContext.current!.cgContext
    context.scaleBy(x: CGFloat(pixels)/1024, y: CGFloat(pixels)/1024)
    let background = NSBezierPath(roundedRect: NSRect(x: 30,y:30,width:964,height:964),xRadius:220,yRadius:220)
    NSColor(calibratedRed:0.025,green:0.060,blue:0.10,alpha:1).setFill()
    background.fill()
    NSGradient(colors:[NSColor(calibratedRed:0.10,green:0.19,blue:0.28,alpha:1),NSColor(calibratedRed:0.015,green:0.025,blue:0.07,alpha:1)])!.draw(in:background,angle:-65)
    context.saveGState()
    background.addClip()
    for arm in 0..<3 {
        for k in 0..<160 {
            let t0 = Double(k)/160, t1 = Double(k+1)/160
            func point(_ t: Double) -> CGPoint {
                let angle = t * 5.6 + Double(arm) * 2 * Double.pi/3
                let radius = 35 + 367 * pow(t,0.78)
                return CGPoint(x:512+radius*cos(angle),y:512+radius*sin(angle))
            }
            let path = NSBezierPath(); path.move(to:point(t0)); path.line(to:point(t1))
            path.lineCapStyle = .round
            path.lineWidth = CGFloat(8+47*sin(Double.pi*t0))
            let hue = 0.47 + 0.27*t0
            NSColor(calibratedHue:hue,saturation:0.70,brightness:0.98,alpha:1).setStroke()
            path.stroke()
        }
    }
    let center = NSBezierPath(ovalIn:NSRect(x:489,y:489,width:46,height:46))
    NSColor(calibratedRed:0.70,green:1,blue:0.98,alpha:1).setFill(); center.fill()
    context.restoreGState()
    image.unlockFocus()
    let bitmap = NSBitmapImageRep(data:image.tiffRepresentation!)!
    let filename = "icon_\(size)x\(size)@\(scale)x.png"
    try bitmap.representation(using:.png,properties:[:])!.write(to:directory.appendingPathComponent(filename))
    entries.append(["size":"\(size)x\(size)","idiom":"mac","filename":filename,"scale":"\(scale)x"])
}
let contents: [String:Any] = ["images":entries,"info":["author":"xcode","version":1]]
try JSONSerialization.data(withJSONObject:contents,options:[.prettyPrinted,.sortedKeys]).write(to:directory.appendingPathComponent("Contents.json"))
