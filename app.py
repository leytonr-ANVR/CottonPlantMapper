
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
    ax.add_patch(Ellipse((x,y), s*.70, s*.85, facecolor="#62ad3c", edgecolor="#347b31", lw=.8, zorder=8))
    for ang in [45,90,135,225,270,315]:
        a=np.deg2rad(ang)
        tip=(x+np.cos(a)*s*1.08,y+np.sin(a)*s*1.12)
        l=(x+np.cos(a+.24)*s*.34,y+np.sin(a+.24)*s*.34)
        r=(x+np.cos(a-.24)*s*.34,y+np.sin(a-.24)*s*.34)
        ax.add_patch(Polygon([l,tip,r],closed=True,facecolor="#30943c",edgecolor="#277533",lw=.5,zorder=9))

def draw_white_flower(ax, x, y, s=.15):
    for ang in [18,90,162,234,306]:
        a=np.deg2rad(ang)
        ax.add_patch(Ellipse((x+np.cos(a)*s*.42,y+np.sin(a)*s*.38),s*1.05,s*.80,angle=ang,
                             facecolor="#fffdfd",edgecolor="#eadde3",lw=.7,zorder=9))
    ax.add_patch(Circle((x,y),s*.24,facecolor="#f3d5df",edgecolor="#d6aeba",lw=.6,zorder=10))
    for ang in [205,240,275,310,345]:
        a=np.deg2rad(ang)
        tip=(x+np.cos(a)*s*1.0,y+np.sin(a)*s*.95)
        l=(x+np.cos(a+.18)*s*.32,y+np.sin(a+.18)*s*.30)
        r=(x+np.cos(a-.18)*s*.32,y+np.sin(a-.18)*s*.30)
        ax.add_patch(Polygon([l,tip,r],closed=True,facecolor="#359a3f",edgecolor="#287532",lw=.5,zorder=8))

def draw_boll(ax, x, y, s=.14):
    for dx,ang in [(-.28,-15),(0,0),(.28,15)]:
        ax.add_patch(Ellipse((x+dx*s,y+.06*s),s*.68,s*1.0,angle=ang,
                             facecolor="#67ad3c",edgecolor="#3e812d",lw=.8,zorder=8))
    for ang in [55,85,115,210,270,330]:
        a=np.deg2rad(ang)
        tip=(x+np.cos(a)*s*1.18,y+np.sin(a)*s*1.22)
        l=(x+np.cos(a+.18)*s*.36,y+np.sin(a+.18)*s*.34)
        r=(x+np.cos(a-.18)*s*.36,y+np.sin(a-.18)*s*.34)
        ax.add_patch(Polygon([l,tip,r],closed=True,facecolor="#4c9433",edgecolor="#31752a",lw=.5,zorder=9))

def draw_cracked_boll(ax, x, y, s=.14):
    # Cracked/open boll: green bracts with a visible white cotton lock.
    draw_boll(ax, x, y, s)
    ax.add_patch(Ellipse((x,y+.05*s),s*.48,s*.62,facecolor="#fffdf8",edgecolor="#d8d8d8",lw=.7,zorder=10))
    ax.plot([x,x],[y-.24*s,y+.34*s],color="#7a522c",lw=.9,zorder=11)

def draw_missing(ax, x, y, s=.12):
    # Brown dashed circle, matching the missing-position symbol in the requested legend.
    ax.add_patch(Circle((x,y),s*.72,fill=False,edgecolor="#8a3b1f",lw=1.5,linestyle=(0,(4,3)),zorder=9))

def draw_symbol(ax, x, y, fruit, s=.13):
    if fruit == "Boll": draw_boll(ax,x,y,s)
    elif fruit == "White Flower": draw_white_flower(ax,x,y,s)
    elif fruit == "Square": draw_square(ax,x,y,s)
    elif fruit == "Cracked Boll": draw_cracked_boll(ax,x,y,s)
    elif fruit == "Missing Fruit": draw_missing(ax,x,y,s)

def legend_image():
    fig, ax = plt.subplots(figsize=(2.9, 4.2))
    ax.set_xlim(0, 3); ax.set_ylim(0, 6); ax.axis("off")
    items = [
        ("Boll", draw_boll),
        ("White Flower", draw_white_flower),
        ("Square", draw_square),
        ("Cracked Boll", draw_cracked_boll),
        ("Missing Fruit", draw_missing),
    ]
    for i, (label, fn) in enumerate(items):
        y = 5.35 - i*1.02
        fn(ax, .55, y, .19 if label != "White Flower" else .20)
        ax.text(1.02, y, label, va="center", ha="left", fontsize=12, color="#0b2e55", fontweight="bold")
    fig.tight_layout(pad=.25)
    return fig

def make_figure(df, min_node, max_node, show_labels=True, show_positions=True, show_ground=True):
    fig_h=max(8,(max_node-min_node+1)*.46)
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
                if show_positions:
                    ax.add_patch(Circle((x,py),.052,facecolor="white",edgecolor="#008f45",lw=1.1,zorder=7))
                if fruit != "-":
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
:root{--navy:#062d57;--green:#078447;--border:#d8e3ec}
.block-container{max-width:1550px;padding-top:.65rem}
[data-testid="stSidebar"]{display:none}
.top-title{font-size:31px;font-weight:800;color:var(--navy)}
.top-sub{font-size:15px;color:#315d8a;margin-top:4px}
.panel,.legend-card{border:1px solid var(--border);border-radius:12px;background:white;padding:12px 14px}
.metric-card{border:1px solid var(--border);border-radius:12px;padding:12px;text-align:center;background:linear-gradient(#fff,#f7fbfb)}
.metric-label{font-size:14px;color:#17395f}.metric-value{font-size:27px;font-weight:800;color:#0b7a38}
.stTabs [data-baseweb="tab"]{font-weight:700;color:var(--navy)}
.stTabs [aria-selected="true"]{color:var(--green)!important}
</style>
""", unsafe_allow_html=True)

logo=Path(__file__).with_name("agnvet_rural_logo.png")
h1,h2,h3=st.columns([1,2.7,2.2],vertical_alignment="center")
with h1:
    if logo.exists(): st.image(str(logo),width=190)
with h2:
    st.markdown('<div class="top-title">Cotton Plant Mapper 🌱</div><div class="top-sub">Accurate mapping. Better decisions.</div>',unsafe_allow_html=True)
with h3:
    e1,e2,e3=st.columns(3); pdfslot=e1.empty(); pngslot=e2.empty(); clear=e3.button("🗑 Clear Map",use_container_width=True)

st.markdown("### 🛡 Report Details")
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

left,centre,right=st.columns([1.22,1.52,.48],gap="medium")
show_labels=show_positions=show_ground=True

with left:
    t1,t2,t3=st.tabs(["📋 Data Entry","☷ Summary","⚙ Settings"])
    with t1:
        st.markdown("### Node Entry")
        a,b,c=st.columns(3); a.info("**V Vegetative**"); b.success("**R Reproductive**"); c.warning("**VL Vegetative Lateral**")
        edited=st.data_editor(
            st.session_state.plant_matrix,use_container_width=True,hide_index=True,height=520,
            column_config={
                "Node":st.column_config.NumberColumn("Node",disabled=True),
                "Node Type":st.column_config.SelectboxColumn("Type",options=NODE_TYPES),
                "Position Count":st.column_config.NumberColumn("Positions",min_value=0,max_value=6,step=1),
                **{f"Position {p}":st.column_config.SelectboxColumn(f"Pos {p}",options=FRUIT_TYPES) for p in range(1,7)}
            },
            column_order=["Node","Node Type","Position 1","Position 2","Position 3","Position Count","Position 4","Position 5","Position 6","Notes"],
            key="editor"
        )
        st.session_state.plant_matrix=normalize_matrix(edited,int(min_node),int(max_node))
    with t2:
        m=calculate_metrics(st.session_state.plant_matrix)
        st.metric("Total Nodes",m["total_nodes"]); st.metric("Total Positions",m["total_positions"])
        st.metric("Held Positions",m["held_positions"]); st.metric("Missing Fruit",m["missing_positions"])
        st.metric("Retention %",f'{m["retention"]:.1f}%' if m["retention"] is not None else "—")
    with t3:
        show_labels=st.toggle("Show Labels",True); show_positions=st.toggle("Show Positions",True); show_ground=st.toggle("Show Ground Line",True)

with centre:
    st.markdown("<h3 style='text-align:center'>Cotton Plant Map</h3>",unsafe_allow_html=True)
    fig=make_figure(st.session_state.plant_matrix,int(min_node),int(max_node),show_labels,show_positions,show_ground)
    st.pyplot(fig,use_container_width=True)
    png=to_bytes(fig,"png"); pdf=to_bytes(fig,"pdf")
    with pdfslot.container(): st.download_button("⇩ Export PDF",pdf,"cotton_plant_map.pdf","application/pdf",use_container_width=True)
    with pngslot.container(): st.download_button("⇩ Export PNG",png,"cotton_plant_map.png","image/png",use_container_width=True)
    plt.close(fig)

with right:
    st.markdown('<div class="legend-card"><h4 style="color:#062d57;margin-top:0">Legend</h4></div>',unsafe_allow_html=True)
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
