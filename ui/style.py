# ui/helpers.py
# Helper functions to render nice HTML snippets inside Gradio (embedded HTML)

def status_box(text: str, kind: str = "success") -> str:
    """
    Render a status box: 'success' | 'error' | 'info'.
    Returns an HTML string suitable for gr.HTML()/gr.Markdown().
    Inputs:
        text (str)
        kind (str)
    Outputs:
        str (HTML)
    """
    styles = {
        "success": ("#ecfdf5", "#10b981", "#065f46", "✅"),
        "error":   ("#fef2f2", "#ef4444", "#7f1d1d", "❌"),
        "info":    ("#eff6ff", "#3b82f6", "#1e3a8a", "ℹ️"),
    }
    bg, border, fg, icon = styles.get(kind, styles["info"])
    return f"""
            <div style="padding:12px 14px;border:1px solid {border}33;background:{bg};
                        color:{fg};border-radius:10px;font-weight:500;">
            <span style="font-size:18px;margin-right:8px">{icon}</span>{text}
            </div>
            """

def metric_card(label: str, value: int | float | str, sub: str = "points in collection") -> str:
    """
    Mini metric card (large number), useful for counters like number of vectors.
    Inputs:
        label (str)
        value (int|float|str)
        sub (str)
    Outputs:
        str (HTML)
    """
    try:
        value_fmt = f"{int(value):,}"
    except Exception:
        value_fmt = str(value)

    return f"""
                <div style="display:flex;gap:12px;align-items:center;padding:12px;
                            border:1px solid #e5e7eb;border-radius:12px;background:#fff;">
                <div style="width:44px;height:44px;border-radius:10px;background:#eef2ff;
                            display:flex;align-items:center;justify-content:center;font-size:22px">📦</div>
                <div>
                    <div style="font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:.06em">{label}</div>
                    <div style="font-size:28px;font-weight:700;line-height:1">{value_fmt}</div>
                    <div style="font-size:12px;color:#9ca3af">{sub}</div>
                </div>
                </div>
                """