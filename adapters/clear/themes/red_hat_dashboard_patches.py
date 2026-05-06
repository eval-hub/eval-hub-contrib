"""String replacements to align CLEAR's embedded dashboard JS with Red Hat styling."""

from __future__ import annotations

_NC_RH = (
    "const nc = cnt => { if(maxC===minC) return '#e0e0e0'; const n=(cnt-minC)/(maxC-minC); "
    "return n>.7?'#151515':n>.4?'#707070':'#bdbdbd'; };"
)
_HEAT_BG_RH = (
    "function heatBg(val, min, max) {\n"
    "  if(max===min) return '#fce3e3';\n"
    "  const t=(val-min)/(max-min);\n"
    "  const r=Math.round(255-t*3), g=Math.round(255-t*28), b=Math.round(255-t*28);\n"
    "  return `rgb(${r},${g},${b})`;\n"
    "}\n"
)
_HEAT_BG_RH_NODE_USAGE = (
    "function heatBg(val, min, max) {\n"
    "  // Blend #ffffff → #fce3e3 (--primary-bg) for Node Usage cells\n"
    "  if(max===min) return '#fce3e3';\n"
    "  const t=(val-min)/(max-min);\n"
    "  const r=Math.round(255-t*3), g=Math.round(255-t*28), b=Math.round(255-t*28);\n"
    "  return `rgb(${r},${g},${b})`;\n"
    "}\n"
)
_HEAT_FG_RH = (
    "function heatFg(val, min, max) {\n"
    "  if(max===min) return '#151515';\n"
    "  const t=(val-min)/(max-min);\n"
    "  return t>0.5 ? '#a60000' : '#151515';\n"
    "}\n"
)

RED_HAT_DASHBOARD_JS_PATCHES: tuple[tuple[str, str], ...] = (
    (
        "const nc = cnt => { if(maxC===minC) return '#A5B4FC'; const n=(cnt-minC)/(maxC-minC); "
        "return n>.7?'#4F46E5':n>.4?'#6366F1':'#818CF8'; };",
        _NC_RH,
    ),
    (
        "const nc = cnt => { if(maxC===minC) return '#FECACA'; const n=(cnt-minC)/(maxC-minC); "
        "return n>.7?'#CC0000':n>.4?'#EE0000':'#F87171'; };",
        _NC_RH,
    ),
    (
        "function heatBg(val, min, max) {\n"
        "  // Returns a background color from light blue to deep indigo based on normalized value\n"
        "  if(max===min) return 'rgba(99,102,241,0.10)';\n"
        "  const t=(val-min)/(max-min);\n"
        "  const r=Math.round(238 - t*139);  // 238 -> 99\n"
        "  const g=Math.round(242 - t*140);  // 242 -> 102\n"
        "  const b=Math.round(255 - t*14);   // 255 -> 241\n"
        "  const a=(0.08 + t*0.25).toFixed(2);\n"
        "  return `rgba(${99},${102},${241},${a})`;\n"
        "}\n",
        _HEAT_BG_RH_NODE_USAGE,
    ),
    (
        "function heatBg(val, min, max) {\n"
        "  // Returns a background color from light red to deep red based on normalized value\n"
        "  if(max===min) return 'rgba(238,0,0,0.10)';\n"
        "  const t=(val-min)/(max-min);\n"
        "  const a=(0.08 + t*0.25).toFixed(2);\n"
        "  return `rgba(238,0,0,${a})`;\n"
        "}\n",
        _HEAT_BG_RH,
    ),
    (
        "function heatBg(val, min, max) {\n"
        "  if(max===min) return 'rgba(238,0,0,0.06)';\n"
        "  const t=(val-min)/(max-min);\n"
        "  const a=(0.06 + t*0.22).toFixed(2);\n"
        "  return `rgba(238,0,0,${a})`;\n"
        "}\n",
        _HEAT_BG_RH,
    ),
    (
        "function heatFg(val, min, max) {\n"
        "  if(max===min) return '#334155';\n"
        "  const t=(val-min)/(max-min);\n"
        "  return t>0.5 ? '#312E81' : '#334155';\n"
        "}\n",
        _HEAT_FG_RH,
    ),
    (
        "function heatFg(val, min, max) {\n"
        "  if(max===min) return '#334155';\n"
        "  const t=(val-min)/(max-min);\n"
        "  return t>0.5 ? '#7F1D1D' : '#334155';\n"
        "}\n",
        _HEAT_FG_RH,
    ),
    ("ctx.strokeStyle='#4338CA';", "ctx.strokeStyle='#151515';"),
    ("ctx.strokeStyle='#CC0000';", "ctx.strokeStyle='#151515';"),
    ("ctx.font='600 13px Inter,sans-serif';", "ctx.font=\"600 13px 'Red Hat Text',sans-serif\";"),
    ("ctx.font='600 12px Inter,sans-serif';", "ctx.font=\"600 12px 'Red Hat Text',sans-serif\";"),
    ("ctx.clearRect(0,0,W,H); ctx.fillStyle='#FAFBFC';", "ctx.clearRect(0,0,W,H); ctx.fillStyle='#f2f2f2';"),
    ('<span style="color:#93C5FD">', '<span style="color:#a60000">'),
    ('<span style="color:#F87171">', '<span style="color:#a60000">'),
    ("word-break:break-word;color:#E2E8F0", "word-break:break-word;color:#e0e0e0"),
    ("ctx.strokeStyle='#94A3B8'", "ctx.strokeStyle='#c7c7c7'"),
    ("ctx.fillStyle='#64748B'", "ctx.fillStyle='#707070'"),
    ("ctx.strokeStyle='#CBD5E1'", "ctx.strokeStyle='#e0e0e0'"),
    ("ctx.fillStyle='#1E293B'", "ctx.fillStyle='#151515'"),
    (
        "let fc='#16A34A'; if(t>=.66) fc='#DC2626'; else if(t>=.33) fc='#D97706';",
        "let fc='#16A34A'; if(t>=.66) fc='#ee0000'; else if(t>=.33) fc='#D97706';",
    ),
    (
        "let fc='#707070'; if(t>=.66) fc='#ee0000'; else if(t>=.33) fc='#a60000';",
        "let fc='#16A34A'; if(t>=.66) fc='#ee0000'; else if(t>=.33) fc='#D97706';",
    ),
    (
        "if(d.severity>=.7){sb='#fce3e3';sf='#a60000';}else if(d.severity>=.4){sb='#f5f5f5';sf='#707070';}else{sb='#f2f2f2';sf='#151515';}",
        "if(d.severity>=.7){sb='#FEE2E2';sf='#991B1B';}else if(d.severity>=.4){sb='#FEF9C3';sf='#854D0E';}else{sb='#D1FAE5';sf='#065F46';}",
    ),
    (
        "<td style=\"font-weight:700;color:#707070\">${d.freq}%</td><td><span class=\"severity-badge\" style=\"background:#f2f2f2;color:#151515\">0.00</span>",
        "<td style=\"font-weight:700;color:#065F46\">${d.freq}%</td><td><span class=\"severity-badge\" style=\"background:#D1FAE5;color:#065F46\">0.00</span>",
    ),
    (
        "<div style=\"text-align:center;padding:20px;color:#707070;\">No issues discovered",
        "<div style=\"text-align:center;padding:20px;color:#10B981;\">No issues discovered",
    ),
    ("<td style=\"font-weight:600;color:#334155\">", "<td style=\"font-weight:600;color:#4d4d4d\">"),
    ("<td style=\"color:#1E293B\">", "<td style=\"color:#151515\">"),
)

__all__ = ["RED_HAT_DASHBOARD_JS_PATCHES"]
