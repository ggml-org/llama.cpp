import Foundation
import os
#if canImport(AppKit)
import AppKit
#endif
#if canImport(UIKit)
import UIKit
#endif

/// Wires every text input in the running app to
/// ``CovertTriggerMonitor`` so a user-chosen phrase typed
/// anywhere fires the wipe silently.
///
/// The design (section 8.3) calls for the trigger to be checked
/// in every text input: chat, library search, materials search,
/// settings fields, the capture floating window, and any
/// third-party text field. The doc explicitly says the explicit
/// `.onChange` calls are the documented contract; the swizzle
/// here is the safety net that catches inputs we forgot to wire
/// (or that arrived from a third-party AppKit view we don't own).
///
/// We use `method_exchangeImplementations` to swap
/// `-[NSTextView didChangeText]` (the AppKit method SwiftUI's
/// `TextField`/`SecureField`/`TextEditor` ultimately calls via
/// its NSTextView backing on macOS). Every keystroke that flows
/// through the NSTextView now ALSO calls our monitor. The swap
/// is invisible to the user; the trigger check happens in
/// microseconds and is a no-op when no phrase is configured.
///
/// iOS: SwiftUI on iOS is backed by `UITextField` /
/// `UITextView`, not `NSTextView`. We install a parallel swizzle
/// on `-[UITextField sendActions:for:]` and on
/// `-[UITextView didChangeSelection]`. The iOS `TextField`/
/// `SecureField` are backed by `UITextField`; SwiftUI's
/// `TextEditor` is backed by `UITextView`; both are covered.
///
/// "Verified manually" applies to the swizzle: XCTest cannot
/// drive a real keystroke through an AppKit/UIKit method
/// swizzle. The unit tests in `CovertTriggerMonitorTests` cover
/// the monitor's matching rules; the swizzle is the production
/// path that delivers `observe(text:)` calls to it.
public enum TextInputInterceptor {
    private static let logger = Logger(
        subsystem: "com.tessera.studio",
        category: "text-input-interceptor"
    )

    /// True if install() has swapped the methods. The composition
    /// root is expected to call this once on launch.
    private static var installed = false

    /// Install the swizzles. Idempotent: a second call is a
    /// no-op. Returns true if the swizzles were installed (or
    /// were already installed), false if a swizzle could not
    /// be installed on the current platform.
    @discardableResult
    public static func install() -> Bool {
        guard !installed else { return true }
        var ok = true
        #if canImport(AppKit)
        if !installNSTextViewSwizzle() { ok = false }
        #endif
        #if canImport(UIKit)
        if !installUITextFieldSwizzle() { ok = false }
        if !installUITextViewSwizzle() { ok = false }
        #endif
        installed = ok
        if ok {
            logger.notice("covert_trigger_text_input_interceptor_installed")
        } else {
            logger.error("covert_trigger_text_input_interceptor_install_failed")
        }
        return ok
    }

    /// Per-keystroke hook called from the swizzled methods.
    /// The monitor's `observe(text:)` is async; we dispatch
    /// to a background task so the caller's hot path (the
    /// AppKit text-change notification) returns immediately.
    /// The wipe is fire-and-forget; the visible UI continues
    /// to render normally.
    fileprivate static func monitorText(_ text: String) {
        Task.detached(priority: .userInitiated) {
            _ = await CovertTriggerMonitor.shared.observe(text: text)
        }
    }
}

#if canImport(AppKit)
private extension NSTextView {
    /// The swizzled implementation of `-[NSTextView didChangeText]`.
    /// The runtime maps this method's IMP to the original
    /// `didChangeText` selector AFTER `installNSTextViewSwizzle`
    /// runs. So at runtime, calling `didChangeText` on an
    /// NSTextView instance calls THIS method; calling
    /// `dchadeSwizzledDidChangeText` calls the original.
    @objc func dchadeSwizzledDidChangeText() {
        // Call the original. After the exchange, the IMP
        // stored under `dchadeSwizzledDidChangeText` is the
        // original AppKit `didChangeText` body, so this
        // recursive-looking call actually runs AppKit's
        // notification + layout logic.
        self.dchadeSwizzledDidChangeText()
        // Forward the current text to the monitor. The
        // `string` accessor is safe to call here because
        // AppKit delivers text-change notifications on
        // the main thread.
        let text = self.string
        TextInputInterceptor.monitorText(text)
    }
}

private func installNSTextViewSwizzle() -> Bool {
    // The original method is `-[NSTextView didChangeText]`,
    // an `@objc` AppKit method visible to Swift via the
    // public header. We grab it by selector to sidestep any
    // Swift-override-on-non-dynamic-method warnings.
    let originalSelector = NSSelectorFromString("didChangeText")
    let swizzledSelector = #selector(NSTextView.dchadeSwizzledDidChangeText)
    guard
        let original = class_getInstanceMethod(NSTextView.self, originalSelector),
        let swizzled = class_getInstanceMethod(NSTextView.self, swizzledSelector)
    else {
        return false
    }
    method_exchangeImplementations(original, swizzled)
    return true
}
#endif

#if canImport(UIKit)
private extension UITextField {
    /// The swizzled implementation of
    /// `-[UITextField sendActions:for:]`. After the exchange,
    /// calling `sendActions:for:` on a UITextField calls this
    /// method; calling this method calls the original.
    @objc func dchadeSwizzledSendActions(for controlEvent: UIControl.Event, for event: UIEvent?) {
        // Run the original first so the responder chain
        // (delegate callbacks, target/action) still fires.
        self.dchadeSwizzledSendActions(for: controlEvent, for: event)
        // Observe on every action; the monitor's substring
        // check is microseconds and short-circuits on text
        // that doesn't contain the phrase.
        let text = self.text ?? ""
        TextInputInterceptor.monitorText(text)
    }
}

private extension UITextView {
    /// The swizzled implementation of
    /// `-[UITextView didChangeSelection]`. UITextView doesn't
    /// have a `didChangeText` method, but it does fire
    /// `didChangeSelection` on every typed character as the
    /// caret moves. This is the closest AppKit-equivalent of
    /// the per-keystroke hook.
    @objc func dchadeSwizzledDidChangeSelection() {
        self.dchadeSwizzledDidChangeSelection()
        let text = self.text ?? ""
        TextInputInterceptor.monitorText(text)
    }
}

private func installUITextFieldSwizzle() -> Bool {
    let originalSelector = NSSelectorFromString("sendActions:for:")
    let swizzledSelector = #selector(UITextField.dchadeSwizzledSendActions(for:for:))
    guard
        let original = class_getInstanceMethod(UITextField.self, originalSelector),
        let swizzled = class_getInstanceMethod(UITextField.self, swizzledSelector)
    else {
        return false
    }
    method_exchangeImplementations(original, swizzled)
    return true
}

private func installUITextViewSwizzle() -> Bool {
    let originalSelector = NSSelectorFromString("didChangeSelection")
    let swizzledSelector = #selector(UITextView.dchadeSwizzledDidChangeSelection)
    guard
        let original = class_getInstanceMethod(UITextView.self, originalSelector),
        let swizzled = class_getInstanceMethod(UITextView.self, swizzledSelector)
    else {
        return false
    }
    method_exchangeImplementations(original, swizzled)
    return true
}
#endif
