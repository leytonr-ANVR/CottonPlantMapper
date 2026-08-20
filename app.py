
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon
from io import BytesIO
from pathlib import Path
import numpy as np

st.set_page_config(page_title="Cotton Plant Mapper", page_icon="🌱", layout="wide")

FRUIT_TYPES = ["-", "Boll", "Cracked Boll", "Square", "White Flower", "Missing Fruit"]
NODE_TYPES = ["Vegetative", "Reproductive", "Vegetative Lateral"]
MAX_POSITIONS = 6

FRUIT_DISPLAY = {
    "-": "—",
    "Boll": "🌿 Boll",
    "Cracked Boll": "◐ Cracked Boll",
    "Square": "🟩 Square",
    "White Flower": "🌸 White Flower",
    "Missing Fruit": "◌ Missing Fruit",
}

TYPE_SHORT = {
    "Vegetative": "V",
    "Reproductive": "R",
    "Vegetative Lateral": "VL",
}

def blank_matrix(min_node=1, max_node=22):
    nodes = list(range(min_node, max_node + 1))
    d = {
        "Node": nodes,
        "Node Type": ["Vegetative" if n <= 7 else "Reproductive" for n in nodes],
        "Position Count": [0 if n <= 7 else 3 for n in nodes],
    }
    for p in range(1, MAX_POSITIONS + 1):
        d[f"Position {p}"] = [
            ("-" if n <= 7 else "Square") if p == 1 else "-"
            for n in nodes
        ]
    d["Notes"] = [""] * len(nodes)
    return pd.DataFrame(d)

def normalize_matrix(df, min_node, max_node):
    base = blank_matrix(min_node, max_node)
    if df is None or df.empty:
        return base
    df = df.copy()
    for c in base.columns:
        if c not in df.columns:
            df[c] = base[c].iloc[0] if len(base) else "-"
    df["Node"] = pd.to_numeric(df["Node"], errors="coerce")
    df = df.dropna(subset=["Node"])
    df["Node"] = df["Node"].astype(int)
    df["Node Type"] = df["Node Type"].where(df["Node Type"].isin(NODE_TYPES), "Reproductive")
    df["Position Count"] = pd.to_numeric(df["Position Count"], errors="coerce").fillna(3).astype(int).clip(0, MAX_POSITIONS)
    for p in range(1, MAX_POSITIONS + 1):
        c = f"Position {p}"
        df[c] = df[c].where(df[c].isin(FRUIT_TYPES), "-")
    keep = list(base.columns)
    out = base.merge(df[keep], on="Node", how="left", suffixes=("", "_saved"))
    for c in keep[1:]:
        s = c + "_saved"
        if s in out.columns:
            out[c] = out[s].fillna(out[c])
    out.loc[out["Node Type"] == "Vegetative", "Position Count"] = 0
    for p in range(1, MAX_POSITIONS + 1):
        out.loc[out["Position Count"] < p, f"Position {p}"] = "-"
    return out[keep]

def calculate_metrics(df):
    total_positions = int(df["Position Count"].sum())
    held = missing = 0
    for _, r in df.iterrows():
        for p in range(1, int(r["Position Count"]) + 1):
            fruit = r[f"Position {p}"]
            if fruit in ["Boll", "Cracked Boll", "Square", "White Flower"]:
                held += 1
            elif fruit == "Missing Fruit":
                missing += 1
    return {
        "total_nodes": len(df),
        "total_positions": total_positions,
        "held_positions": held,
        "missing_positions": missing,
        "retention": held / total_positions * 100 if total_positions else None,
    }

# --- Cotton symbols used by BOTH the map and the legend ---
def draw_square(ax, x, y, s=.12):
    # Compact pointed cotton square/bud.
    ax.add_patch(Ellipse(
        (x,y),s*.66,s*.78,
        facecolor="#62ad3c",edgecolor="#347b31",lw=.8,zorder=9
    ))
    for ang in [0,45,90,135,180,225,270,315]:
        a=np.deg2rad(ang)
        tip=(x+np.cos(a)*s*1.30,y+np.sin(a)*s*1.30)
        l=(x+np.cos(a+.20)*s*.30,y+np.sin(a+.20)*s*.30)
        r=(x+np.cos(a-.20)*s*.30,y+np.sin(a-.20)*s*.30)
        ax.add_patch(Polygon(
            [l,tip,r],closed=True,
            facecolor="#248d38",edgecolor="#1f7130",lw=.5,zorder=10
        ))
    ax.plot([x,x],[y-.48*s,y-.92*s],color="#347d2e",lw=1.0,zorder=7)


def draw_white_flower(ax, x, y, s=.15):
    # Large white cotton flower with a pale pink centre and green bracts.
    petals = [
        (-.42,.10,18), (-.18,.40,62), (.18,.40,118),
        (.43,.10,162), (.25,-.18,215), (-.25,-.18,325)
    ]
    for dx,dy,ang in petals:
        ax.add_patch(Ellipse(
            (x+dx*s,y+dy*s), s*.95,s*.72,angle=ang,
            facecolor="#fffdfd",edgecolor="#eadde5",lw=.75,zorder=9
        ))
    ax.add_patch(Circle(
        (x,y+.02*s),s*.22,
        facecolor="#f1cbd8",edgecolor="#d8a8b8",lw=.6,zorder=11
    ))
    for ang in [205,240,275,310,345]:
        a=np.deg2rad(ang)
        tip=(x+np.cos(a)*s*1.18,y+np.sin(a)*s*1.05)
        l=(x+np.cos(a+.17)*s*.34,y+np.sin(a+.17)*s*.32)
        r=(x+np.cos(a-.17)*s*.34,y+np.sin(a-.17)*s*.32)
        ax.add_patch(Polygon(
            [l,tip,r],closed=True,
            facecolor="#31943b",edgecolor="#26752f",lw=.5,zorder=8
        ))
    ax.plot([x,x],[y-.48*s,y-.92*s],color="#347d2e",lw=1.1,zorder=7)


def draw_boll(ax, x, y, s=.14):
    # Full green cotton boll with pointed bracts.
    for dx, ang, scale in [(-.25,-14,.90),(0,0,1.05),(.25,14,.90)]:
        ax.add_patch(Ellipse(
            (x+dx*s, y+.08*s), s*.78*scale, s*1.12,
            angle=ang, facecolor="#55a630", edgecolor="#2f762b",
            lw=.9, zorder=9
        ))
    # central seam
    ax.plot([x,x],[y-.30*s,y+.48*s],color="#43872f",lw=.7,zorder=10)
    # large bracts around boll
    for ang in [42,72,108,138,218,270,322]:
        a=np.deg2rad(ang)
        tip=(x+np.cos(a)*s*1.38,y+np.sin(a)*s*1.42)
        l=(x+np.cos(a+.20)*s*.40,y+np.sin(a+.20)*s*.38)
        r=(x+np.cos(a-.20)*s*.40,y+np.sin(a-.20)*s*.38)
        ax.add_patch(Polygon(
            [l,tip,r], closed=True,
            facecolor="#2f963a", edgecolor="#24752f",
            lw=.55, zorder=10
        ))
    # short stem
    ax.plot([x,x],[y-.58*s,y-.95*s],color="#347d2e",lw=1.2,zorder=7)


def draw_cracked_boll(ax, x, y, s=.14):
    # Open/cracked boll with obvious white cotton locks.
    for dx in [-.26,0,.26]:
        ax.add_patch(Ellipse(
            (x+dx*s,y+.08*s),s*.56,s*.72,
            facecolor="#fffdf8",edgecolor="#d9d7d0",lw=.7,zorder=11
        ))
    # green shell/bracts around cotton
    for ang in [28,68,112,152,220,270,320]:
        a=np.deg2rad(ang)
        tip=(x+np.cos(a)*s*1.35,y+np.sin(a)*s*1.38)
        l=(x+np.cos(a+.20)*s*.38,y+np.sin(a+.20)*s*.36)
        r=(x+np.cos(a-.20)*s*.38,y+np.sin(a-.20)*s*.36)
        ax.add_patch(Polygon(
            [l,tip,r],closed=True,
            facecolor="#35933b",edgecolor="#26742f",lw=.55,zorder=10
        ))
    ax.plot([x,x],[y-.45*s,y-.95*s],color="#704526",lw=1.2,zorder=8)


def draw_missing(ax, x, y, s=.12):
    # Brown dashed circle, matching the missing-position symbol in the requested legend.
    ax.add_patch(Circle((x,y),s*.72,fill=False,edgecolor="#8a3b1f",lw=1.5,linestyle=(0,(4,3)),zorder=9))

def draw_empty_position(ax, x, y, s=.12):
    # Open green circle used for an empty/available fruiting position.
    ax.add_patch(
        Circle(
            (x, y), s*.70,
            facecolor="white",
            edgecolor="#087a35",
            lw=1.5,
            zorder=9
        )
    )

def draw_symbol(ax, x, y, fruit, s=.13):
    if fruit == "Boll": draw_boll(ax,x,y,s)
    elif fruit == "White Flower": draw_white_flower(ax,x,y,s)
    elif fruit == "Square": draw_square(ax,x,y,s)
    elif fruit == "Cracked Boll": draw_cracked_boll(ax,x,y,s)
    elif fruit == "Missing Fruit": draw_missing(ax,x,y,s)
    elif fruit == "-": draw_empty_position(ax,x,y,s)

def legend_image():
    fig, ax = plt.subplots(figsize=(3.0, 3.55))
    ax.set_xlim(0, 3.25)
    ax.set_ylim(0, 6.55)
    ax.axis("off")

    ax.text(.12, 6.20, "Legend", ha="left", va="center",
            fontsize=13, fontweight="bold", color="#14213d")

    items = [
        ("Boll", draw_boll, .24),
        ("White Flower", draw_white_flower, .24),
        ("Square", draw_square, .21),
        ("Cracked Boll", draw_cracked_boll, .24),
        ("Missing Fruit", draw_missing, .20),
        ("Position (Empty)", draw_empty_position, .20),
    ]

    y0 = 5.45
    for i, (label, fn, size) in enumerate(items):
        y = y0 - i*.91
        fn(ax, .48, y, size)
        ax.text(.98, y, label, va="center", ha="left",
                fontsize=10.8, color="#111111")

    fig.subplots_adjust(left=.02,right=.98,top=.99,bottom=.01)
    return fig

def make_figure(df, min_node, max_node, show_labels=True, show_positions=True, show_ground=True):
    node_span = max_node - min_node + 1
    fig_h = 7.35 if node_span <= 24 else 8.25
    fig,ax=plt.subplots(figsize=(7.4,fig_h))
    ground=min_node-1.0
    ax.plot([0,0],[ground,max_node+.8],color="#008f45",lw=4,zorder=2)
    if show_ground:
        ax.plot([-2.25,2.25],[ground,ground],color="#9b5e16",lw=5,solid_capstyle="round")
    ax.add_patch(Polygon([(-.18,ground),(0,ground+.28),(.18,ground)],closed=True,facecolor="#008f45",edgecolor="#008f45"))

    for node in range(min_node,max_node+1):
        row=df[df["Node"]==node]
        if row.empty: continue
        row=row.iloc[0]
        typ=row["Node Type"]
        side=-1 if node%2 else 1
        y=node
        count=int(row["Position Count"])

        if typ=="Vegetative":
            ex=side*.86
            ax.plot([0,ex],[y,y+.20],color="#008f45",lw=2.3)
            ax.add_patch(Ellipse((ex+side*.14,y+.24),.30,.14,angle=20*side,facecolor="#62ae45",edgecolor="#39863a",lw=.6))
        else:
            effective=0
            for p in range(1,count+1):
                if row[f"Position {p}"] != "-":
                    effective=p
            dcount=max(effective,1 if count else 0)
            bl=.64+max(0,dcount-1)*.40
            if typ=="Vegetative Lateral":
                ax.plot([0,side*.45,side*bl],[y,y-.10,y-.34],color="#008f45",lw=2.4)
                coords=[(p,side*(.27+(bl-.27)*(p/(effective+1))),y-.05-.24*(p/(effective+1))) for p in range(1,effective+1)]
            else:
                ax.plot([0,side*.44,side*bl],[y,y+.04,y+.22],color="#008f45",lw=2.4)
                coords=[(p,side*(.28+(bl-.28)*(p/max(effective,1))),y+.03+.18*(p/max(effective,1))) for p in range(1,effective+1)]
            for p,x,py in coords:
                fruit=row[f"Position {p}"]
                if fruit == "-" and show_positions:
                    draw_empty_position(ax, x, py, .10)
                elif fruit != "-":
                    draw_symbol(ax,x,py,fruit,.13)
                if show_labels:
                    ax.text(x+(0.07 if side>0 else -0.07),py+.10,f"{node}-{p}",fontsize=6.5,
                            ha="left" if side>0 else "right",color="#334")
        if show_labels:
            mark={"Vegetative":"V","Reproductive":"R","Vegetative Lateral":"VL"}[typ]
            ax.text(.09 if side<0 else -.09,y,f"{node} {mark}",fontsize=7.5,fontweight="bold",
                    ha="left" if side<0 else "right",va="center")

    ax.text(0,max_node+.95,"Terminal",ha="center",fontsize=11,fontweight="bold")
    ax.plot([0,-.15],[max_node+.78,max_node+1.02],color="#4da83b",lw=2.3)
    ax.plot([0,.15],[max_node+.78,max_node+1.02],color="#4da83b",lw=2.3)
    ax.set_xlim(-2.4,2.4); ax.set_ylim(ground-.12,max_node+1.3); ax.axis("off")
    fig.tight_layout()
    return fig

def to_bytes(fig, fmt):
    b=BytesIO(); fig.savefig(b,format=fmt,dpi=220 if fmt=="png" else None,bbox_inches="tight"); b.seek(0); return b

st.markdown("""
<style>
:root{
  --navy:#062d57;
  --green:#078447;
  --border:#d8e3ec;
  --page:#ffffff;
}

/* Keep the whole app comfortably inside a normal desktop browser viewport. */
.block-container{
  width:min(96vw, 1440px) !important;
  max-width:1440px !important;
  padding:0.45rem 0.8rem 1rem !important;
  margin:0 auto !important;
}
[data-testid="stSidebar"]{display:none}

html, body, [class*="css"]{
  font-size:15px;
}

.top-title{
  font-size:clamp(24px,2vw,31px);
  font-weight:800;
  color:var(--navy);
  line-height:1.05;
  white-space:nowrap;
}
.top-sub{
  font-size:clamp(12px,1vw,15px);
  color:#315d8a;
  margin-top:4px;
}

.panel,.legend-card{
  border:1px solid var(--border);
  border-radius:12px;
  background:#fff;
  padding:10px 12px;
}

.metric-card{
  border:1px solid var(--border);
  border-radius:12px;
  padding:10px 8px;
  text-align:center;
  background:linear-gradient(#fff,#f7fbfb);
  min-height:82px;
}
.metric-label{font-size:13px;color:#17395f}
.metric-value{font-size:24px;font-weight:800;color:#0b7a38;margin-top:3px}

/* Tighter controls so the top report row does not dominate the page. */
[data-testid="stTextInput"] label,
[data-testid="stDateInput"] label,
[data-testid="stNumberInput"] label{
  font-size:12px !important;
}
[data-baseweb="input"]{
  min-height:38px !important;
}

.stTabs [data-baseweb="tab-list"]{gap:2px}
.stTabs [data-baseweb="tab"]{
  font-weight:700;
  color:var(--navy);
  padding-left:10px;
  padding-right:10px;
}
.stTabs [aria-selected="true"]{color:var(--green)!important}

/* Make the table and chart fill their panels without overflowing them. */
div[data-testid="stDataEditor"]{
  width:100% !important;
  border:1px solid var(--border);
  border-radius:8px;
  overflow:hidden;
}
div[data-testid="stImage"],
div[data-testid="stImage"] img,
div[data-testid="stPyplot"] img{
  max-width:100% !important;
  height:auto !important;
}

.stButton>button,.stDownloadButton>button{
  border-radius:8px!important;
  font-weight:700!important;
  min-height:38px;
  padding:.35rem .55rem;
  font-size:13px;
}

/* Desktop scaling */
@media (max-width: 1400px){
  .block-container{width:98vw !important;padding-left:.55rem !important;padding-right:.55rem !important}
  html, body, [class*="css"]{font-size:14px}
  .top-title{font-size:26px}
  .metric-value{font-size:22px}
}

/* Smaller laptops: keep everything readable and reduce whitespace. */
@media (max-width: 1180px){
  .top-title{font-size:23px;white-space:normal}
  .top-sub{font-size:12px}
  .metric-label{font-size:12px}
  .metric-value{font-size:20px}
  .stButton>button,.stDownloadButton>button{font-size:12px}
}

.node-head{
  display:grid;
  grid-template-columns:38px 115px 1.18fr 1.02fr 1.02fr;
  gap:6px;
  align-items:center;
  padding:5px 6px 3px;
  border-bottom:1px solid #dfe7ee;
  color:#17395f;
  font-size:12px;
  font-weight:700;
  text-align:center;
}
.node-row{
  display:grid;
  grid-template-columns:38px 115px 1.18fr 1.02fr 1.02fr;
  gap:6px;
  align-items:center;
  padding:4px 6px;
  border-bottom:1px solid #edf2f6;
}
.node-num{
  text-align:center;
  font-weight:700;
  color:#132f52;
}
.type-pill{
  margin:auto;
  width:30px;
  height:28px;
  border-radius:6px;
  display:flex;
  align-items:center;
  justify-content:center;
  font-weight:800;
  font-size:13px;
}
.type-r{background:#e8f6ed;color:#19833e}
.type-v{background:#eaf2ff;color:#2b6fc5}
.type-vl{background:#fff0e8;color:#ef6c21}
.compact-entry [data-testid="stSelectbox"] label{display:none}
.compact-entry [data-baseweb="select"]>div{
  min-height:34px!important;
  height:34px!important;
  font-size:12px!important;
}
.compact-entry [data-testid="stSelectbox"]{
  margin-bottom:0!important;
}


.compact-entry [data-testid="stSelectbox"] div[data-baseweb="select"]{
  width:100% !important;
}


/* Node Type dropdown: enough room to show the type currently selected. */
.compact-entry [data-testid="stSelectbox"] [data-baseweb="select"] > div{
    white-space:nowrap !important;
    overflow:hidden !important;
}
.compact-entry [data-testid="stSelectbox"] [data-baseweb="select"] span{
    font-size:12px !important;
    font-weight:600 !important;
}

</style>
""", unsafe_allow_html=True)

logo=Path(__file__).with_name("agnvet_rural_logo.png")
h1,h2,h3=st.columns([1,2.7,2.2],vertical_alignment="center")
with h1:
    if logo.exists(): st.image(str(logo),width=170)
with h2:
    st.markdown('<div class="top-title">Cotton Plant Mapper 🌱</div><div class="top-sub">Accurate mapping. Better decisions.</div>',unsafe_allow_html=True)
with h3:
    e1,e2,e3=st.columns(3); pdfslot=e1.empty(); pngslot=e2.empty(); clear=e3.button("🗑 Clear Map",use_container_width=True)

st.markdown("#### 🛡 Report Details")
r1,r2,r3,r4,r5,r6=st.columns([1.4,1.4,1.2,.85,.7,.7])
farm=r1.text_input("Farm")
paddock=r2.text_input("Paddock Name")
grower=r3.text_input("Grower")
date=r4.date_input("Date")
min_node=r5.number_input("Lowest Node",1,50,1,1)
max_node=r6.number_input("Max Nodes",int(min_node),60,max(22,int(min_node)),1)

if "plant_matrix" not in st.session_state:
    st.session_state.plant_matrix=blank_matrix(int(min_node),int(max_node))
st.session_state.plant_matrix=normalize_matrix(st.session_state.plant_matrix,int(min_node),int(max_node))
if clear:
    st.session_state.plant_matrix=blank_matrix(int(min_node),int(max_node)); st.rerun()

left,centre,right=st.columns([1.18,1.48,.52],gap="small")
show_labels=show_positions=show_ground=True

with left:
    t1,t2,t3=st.tabs(["📋 Data Entry","☷ Summary","⚙ Settings"])
    with t1:
        st.markdown("### Node Entry")

        # Node type selector strip styled like the supplied reference.
        a,b,c=st.columns(3)
        a.markdown('<div style="border:1px solid #cfe0f7;background:#eef5ff;border-radius:8px;padding:8px;text-align:center;color:#1f65b5;font-weight:700;">V &nbsp; Vegetative</div>', unsafe_allow_html=True)
        b.markdown('<div style="border:1px solid #bfe3cb;background:#eefaf2;border-radius:8px;padding:8px;text-align:center;color:#19833e;font-weight:700;">R &nbsp; Reproductive</div>', unsafe_allow_html=True)
        c.markdown('<div style="border:1px solid #ffd5c0;background:#fff5ef;border-radius:8px;padding:8px;text-align:center;color:#ef6c21;font-weight:700;">VL &nbsp; Vegetative Lateral</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="node-head"><div>Node</div><div>Type</div><div>Pos 1</div><div>Pos 2</div><div>Pos 3</div></div>',
            unsafe_allow_html=True
        )

        df_edit = st.session_state.plant_matrix.copy()
        display_df = df_edit.sort_values("Node", ascending=False)

        st.markdown('<div class="compact-entry">', unsafe_allow_html=True)

        for idx, row in display_df.iterrows():
            node = int(row["Node"])
            cnode, ctype, cp1, cp2, cp3 = st.columns([0.38, 1.15, 1.18, 1.02, 1.02], gap="small")

            with cnode:
                st.markdown(
                    f'<div class="node-num" style="padding-top:9px;">{node}</div>',
                    unsafe_allow_html=True
                )

            with ctype:
                current_type = row["Node Type"]
                new_type = st.selectbox(
                    f"Node {node} Type",
                    options=NODE_TYPES,
                    index=NODE_TYPES.index(current_type) if current_type in NODE_TYPES else 1,
                    format_func=lambda x: (
                        "V  Vegetative" if x == "Vegetative"
                        else "R  Reproductive" if x == "Reproductive"
                        else "VL  Vegetative Lateral"
                    ),
                    key=f"node_{node}_type",
                    label_visibility="collapsed"
                )
                df_edit.loc[df_edit["Node"] == node, "Node Type"] = new_type

                # Apply sensible defaults when changing node type.
                if new_type == "Vegetative":
                    df_edit.loc[df_edit["Node"] == node, "Position Count"] = 0
                    for p in range(1, MAX_POSITIONS + 1):
                        df_edit.loc[df_edit["Node"] == node, f"Position {p}"] = "-"
                elif current_type == "Vegetative" and new_type in ["Reproductive", "Vegetative Lateral"]:
                    df_edit.loc[df_edit["Node"] == node, "Position Count"] = 3
                    if new_type == "Reproductive":
                        df_edit.loc[df_edit["Node"] == node, "Position 1"] = "Square"

            # Keep row interaction compact: each visible position is a dropdown.
            visible_positions = []
            for p, col in zip([1,2,3], [cp1,cp2,cp3]):
                with col:
                    current = row[f"Position {p}"]
                    new_value = st.selectbox(
                        f"Node {node} Pos {p}",
                        options=FRUIT_TYPES,
                        index=FRUIT_TYPES.index(current) if current in FRUIT_TYPES else 0,
                        format_func=lambda x: FRUIT_DISPLAY[x],
                        key=f"node_{node}_pos_{p}",
                        label_visibility="collapsed"
                    )
                    visible_positions.append(new_value)

            # Apply visible position edits.
            for p, val in zip([1,2,3], visible_positions):
                df_edit.loc[df_edit["Node"] == node, f"Position {p}"] = val

            # Automatically keep Position Count aligned with the last visible active position,
            # without removing access to positions 4-6 elsewhere in the data model.
            visible_last = 0
            for p in [1,2,3]:
                if df_edit.loc[df_edit["Node"] == node, f"Position {p}"].iloc[0] != "-":
                    visible_last = p

            current_count = int(df_edit.loc[df_edit["Node"] == node, "Position Count"].iloc[0])
            selected_type = df_edit.loc[df_edit["Node"] == node, "Node Type"].iloc[0]
            if selected_type == "Vegetative":
                df_edit.loc[df_edit["Node"] == node, "Position Count"] = 0
            elif visible_last > current_count:
                df_edit.loc[df_edit["Node"] == node, "Position Count"] = visible_last

        st.markdown('</div>', unsafe_allow_html=True)

        st.session_state.plant_matrix = normalize_matrix(df_edit, int(min_node), int(max_node))

        b1,b2,b3 = st.columns(3)
        if b1.button(f"＋ Add Node Below {int(st.session_state.plant_matrix['Node'].max())}", use_container_width=True):
            st.info("Increase **Max Nodes** above to add another node.")
        if b2.button("− Remove Last Node", use_container_width=True):
            st.info("Reduce **Max Nodes** above to remove the last node.")
        if b3.button("🪄 Auto Fill", use_container_width=True):
            df = st.session_state.plant_matrix.copy()
            for i,row in df.iterrows():
                if row["Node Type"] == "Reproductive" and int(row["Position Count"]) > 0 and row["Position 1"] == "-":
                    df.at[i,"Position 1"] = "Square"
            st.session_state.plant_matrix = df
            st.rerun()

    with t2:
        m=calculate_metrics(st.session_state.plant_matrix)
        st.metric("Total Nodes",m["total_nodes"]); st.metric("Total Positions",m["total_positions"])
        st.metric("Held Positions",m["held_positions"]); st.metric("Missing Fruit",m["missing_positions"])
        st.metric("Retention %",f'{m["retention"]:.1f}%' if m["retention"] is not None else "—")
    with t3:
        show_labels=st.toggle("Show Labels",True); show_positions=st.toggle("Show Positions",True); show_ground=st.toggle("Show Ground Line",True)

with centre:
    st.markdown("<h3 style='text-align:center;margin:0 0 2px'>Cotton Plant Map</h3>",unsafe_allow_html=True)
    fig=make_figure(st.session_state.plant_matrix,int(min_node),int(max_node),show_labels,show_positions,show_ground)
    st.pyplot(fig,use_container_width=True)
    png=to_bytes(fig,"png"); pdf=to_bytes(fig,"pdf")
    with pdfslot.container(): st.download_button("⇩ Export PDF",pdf,"cotton_plant_map.pdf","application/pdf",use_container_width=True)
    with pngslot.container(): st.download_button("⇩ Export PNG",png,"cotton_plant_map.png","image/png",use_container_width=True)
    plt.close(fig)

with right:
    lf=legend_image()
    st.pyplot(lf,use_container_width=True)
    plt.close(lf)
    st.markdown('<div class="legend-card"><h4 style="color:#062d57;margin-top:0">Node Types</h4>🟢 <b>R</b> Reproductive<br><br>🔵 <b>V</b> Vegetative<br><br>🟠 <b>VL</b> Vegetative Lateral</div>',unsafe_allow_html=True)

m=calculate_metrics(st.session_state.plant_matrix)
cols=st.columns(5)
for col,(lab,val) in zip(cols,[
    ("🌱 Total Nodes",m["total_nodes"]),("○ Total Positions",m["total_positions"]),
    ("🌿 Held Positions",m["held_positions"]),("◌ Missing Fruit",m["missing_positions"]),
    ("▥ Retention %",f'{m["retention"]:.1f}%' if m["retention"] is not None else "—")
]):
    col.markdown(f'<div class="metric-card"><div class="metric-label">{lab}</div><div class="metric-value">{val}</div></div>',unsafe_allow_html=True)
