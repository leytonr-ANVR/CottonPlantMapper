
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
    data = {
        "Node": nodes,
        "Node Type": ["Vegetative" if n <= 7 else "Reproductive" for n in nodes],
        "Position Count": [0 if n <= 7 else 3 for n in nodes],
    }
    for p in range(1, MAX_POSITIONS + 1):
        if p == 1:
            data[f"Position {p}"] = ["-" if n <= 7 else "Square" for n in nodes]
        else:
            data[f"Position {p}"] = ["-"] * len(nodes)
    data["Notes"] = [""] * len(nodes)
    return pd.DataFrame(data)

def normalize_matrix(df, min_node, max_node):
    base = blank_matrix(min_node, max_node)
    if df is None or df.empty or "Node" not in df.columns:
        return base
    df = df.copy()
    df["Node"] = pd.to_numeric(df["Node"], errors="coerce")
    df = df.dropna(subset=["Node"])
    df["Node"] = df["Node"].astype(int)

    if "Node Type" not in df.columns:
        df["Node Type"] = "Reproductive"
    df["Node Type"] = df["Node Type"].where(df["Node Type"].isin(NODE_TYPES), "Reproductive")

    if "Position Count" not in df.columns:
        df["Position Count"] = 3
    df["Position Count"] = pd.to_numeric(df["Position Count"], errors="coerce").fillna(3).astype(int).clip(0, MAX_POSITIONS)

    pos_cols = [f"Position {p}" for p in range(1, MAX_POSITIONS + 1)]
    for col in pos_cols:
        if col not in df.columns:
            df[col] = "-"
        df[col] = df[col].where(df[col].isin(FRUIT_TYPES), "-")
    if "Notes" not in df.columns:
        df["Notes"] = ""

    keep = ["Node", "Node Type", "Position Count"] + pos_cols + ["Notes"]
    out = base.merge(df[keep], on="Node", how="left", suffixes=("", "_saved"))
    for col in keep[1:]:
        saved = f"{col}_saved"
        if saved in out.columns:
            out[col] = out[saved].fillna(out[col])

    out.loc[out["Node Type"] == "Vegetative", "Position Count"] = 0
    for p in range(1, MAX_POSITIONS + 1):
        out.loc[out["Position Count"] < p, f"Position {p}"] = "-"
    return out[keep]

def matrix_to_long(df):
    rows = []
    for _, row in df.iterrows():
        for p in range(1, int(row["Position Count"]) + 1):
            rows.append({
                "Node": int(row["Node"]),
                "Node Type": row["Node Type"],
                "Position": p,
                "Fruit": row[f"Position {p}"],
            })
    return pd.DataFrame(rows, columns=["Node","Node Type","Position","Fruit"])

def calculate_metrics(df):
    long_df = matrix_to_long(df)
    held = 0 if long_df.empty else int(long_df["Fruit"].isin(["Boll","Cracked Boll","Square","White Flower"]).sum())
    missing = 0 if long_df.empty else int((long_df["Fruit"] == "Missing Fruit").sum())
    total_positions = int(df["Position Count"].sum())
    retention = (held / total_positions * 100) if total_positions else None
    return {
        "total_nodes": len(df),
        "total_positions": total_positions,
        "held_positions": held,
        "missing_positions": missing,
        "retention": retention,
    }

def draw_square(ax, x, y, s=.12):
    ax.add_patch(Ellipse((x,y), s*.75, s*.9, facecolor="#55a630", edgecolor="#2f7d32", lw=.8, zorder=8))
    for ang in [45,90,135,225,270,315]:
        a=np.deg2rad(ang)
        tip=(x+np.cos(a)*s*1.1,y+np.sin(a)*s*1.15)
        l=(x+np.cos(a+.25)*s*.36,y+np.sin(a+.25)*s*.36)
        r=(x+np.cos(a-.25)*s*.36,y+np.sin(a-.25)*s*.36)
        ax.add_patch(Polygon([l,tip,r], closed=True, facecolor="#2f9e44", edgecolor="#267a35", lw=.5,zorder=9))

def draw_flower(ax,x,y,s=.15):
    for ang in [18,90,162,234,306]:
        a=np.deg2rad(ang)
        ax.add_patch(Ellipse((x+np.cos(a)*s*.42,y+np.sin(a)*s*.38),s*1.05,s*.8,angle=ang,
                             facecolor="white",edgecolor="#eadde3",lw=.7,zorder=9))
    ax.add_patch(Circle((x,y),s*.24,facecolor="#f3d6df",edgecolor="#d7aebb",lw=.6,zorder=10))
    for ang in [205,240,275,310,345]:
        a=np.deg2rad(ang)
        tip=(x+np.cos(a)*s*1.0,y+np.sin(a)*s*.95)
        l=(x+np.cos(a+.18)*s*.32,y+np.sin(a+.18)*s*.30)
        r=(x+np.cos(a-.18)*s*.32,y+np.sin(a-.18)*s*.30)
        ax.add_patch(Polygon([l,tip,r], closed=True, facecolor="#2f9e44",edgecolor="#267a35",lw=.5,zorder=8))

def draw_boll(ax,x,y,s=.14):
    for dx,ang in [(-.28,-15),(0,0),(.28,15)]:
        ax.add_patch(Ellipse((x+dx*s,y+.06*s),s*.68,s*1.0,angle=ang,facecolor="#67ad3c",edgecolor="#3e812d",lw=.8,zorder=8))
    for ang in [55,85,115,210,270,330]:
        a=np.deg2rad(ang)
        tip=(x+np.cos(a)*s*1.18,y+np.sin(a)*s*1.22)
        l=(x+np.cos(a+.18)*s*.36,y+np.sin(a+.18)*s*.34)
        r=(x+np.cos(a-.18)*s*.36,y+np.sin(a-.18)*s*.34)
        ax.add_patch(Polygon([l,tip,r],closed=True,facecolor="#4c9433",edgecolor="#31752a",lw=.5,zorder=9))

def draw_cracked(ax,x,y,s=.14):
    draw_boll(ax,x,y,s)
    ax.add_patch(Ellipse((x,y+.05*s),s*.45,s*.55,facecolor="white",edgecolor="#ddd",lw=.6,zorder=10))
    ax.plot([x,x],[y-.25*s,y+.34*s],color="#7a522c",lw=.9,zorder=11)

def draw_missing(ax,x,y,s=.12):
    ax.add_patch(Circle((x,y),s*.7,fill=False,edgecolor="#8a3b1f",lw=1.4,linestyle=(0,(4,3)),zorder=9))

def draw_symbol(ax,x,y,fruit,s=.13):
    if fruit=="Boll": draw_boll(ax,x,y,s)
    elif fruit=="Cracked Boll": draw_cracked(ax,x,y,s)
    elif fruit=="Square": draw_square(ax,x,y,s)
    elif fruit=="White Flower": draw_flower(ax,x,y,s)
    elif fruit=="Missing Fruit": draw_missing(ax,x,y,s)

def make_figure(matrix_df, min_node, max_node, farm, paddock, grower, report_date, show_labels=True, show_positions=True, show_ground=True):
    metrics = calculate_metrics(matrix_df)
    fig_h=max(8.5,(max_node-min_node+1)*.48)
    fig,ax=plt.subplots(figsize=(7.8,fig_h))
    ax.set_facecolor("white")
    ground=min_node-1.05
    ax.plot([0,0],[ground,max_node+.8],color="#008f45",lw=4,zorder=2)

    if show_ground:
        ax.plot([-2.3,2.3],[ground,ground],color="#9b5e16",lw=5,solid_capstyle="round",zorder=1)

    ax.add_patch(Polygon([(-.18,ground),(0,ground+.28),(.18,ground)],closed=True,facecolor="#008f45",edgecolor="#008f45"))

    for node in range(min_node,max_node+1):
        row=matrix_df[matrix_df["Node"]==node]
        if row.empty: continue
        row=row.iloc[0]
        ntype=row["Node Type"]
        side=-1 if node%2 else 1
        y=node

        node_color={"Vegetative":"#4d8ed8","Reproductive":"#0a9b43","Vegetative Lateral":"#f07d18"}[ntype]
        ax.add_patch(Circle((0,y),.065,facecolor="white",edgecolor=node_color,lw=1.5,zorder=10))

        if ntype=="Vegetative":
            end=side*.88
            ax.plot([0,end],[y,y+.22],color="#008f45",lw=2.3,solid_capstyle="round")
            ax.add_patch(Ellipse((end+side*.14,y+.26),.30,.14,angle=20*side,facecolor="#62ae45",edgecolor="#39863a",lw=.6))
            coords=[]
        else:
            count=int(row["Position Count"])
            effective=0
            for p in range(1,count+1):
                if row[f"Position {p}"]!="-":
                    effective=p
            draw_count=max(effective,1 if count>0 else 0)
            branch_len=.65+max(0,draw_count-1)*.40

            if ntype=="Vegetative Lateral":
                ex=side*branch_len
                ey=y-.34
                ax.plot([0,side*.48,ex],[y,y-.1,ey],color="#008f45",lw=2.4,solid_capstyle="round")
                coords=[]
                for p in range(1,effective+1):
                    frac=p/(effective+1)
                    coords.append((p,side*(.26+(branch_len-.26)*frac),y-.05-.24*frac))
            else:
                ex=side*branch_len
                ey=y+.22+max(0,draw_count-2)*.04
                ax.plot([0,side*.46,ex],[y,y+.04,ey],color="#008f45",lw=2.4,solid_capstyle="round")
                coords=[]
                for p in range(1,effective+1):
                    frac=p/max(effective,1)
                    coords.append((p,side*(.28+(branch_len-.28)*frac),y+.03+.18*frac))

            for p,x,py in coords:
                fruit=row[f"Position {p}"]
                if fruit!="-":
                    draw_symbol(ax,x,py,fruit,.13)
                if show_positions:
                    ax.add_patch(Circle((x,py),.055,facecolor="white",edgecolor="#008f45",lw=1.2,zorder=7))
                if show_labels:
                    ax.text(x+(0.08 if side>0 else -0.08),py+.10,f"{node}-{p}",fontsize=6.6,
                            ha="left" if side>0 else "right",color="#334")

        if show_labels:
            mark={"Vegetative":"V","Reproductive":"R","Vegetative Lateral":"VL"}[ntype]
            ax.text(.10 if side<0 else -.10,y,f"{node} {mark}",fontsize=7.5,fontweight="bold",
                    ha="left" if side<0 else "right",va="center",color="#111")

    ax.text(0,max_node+.95,"Terminal",ha="center",va="center",fontsize=11,fontweight="bold")
    ax.plot([0,-.15],[max_node+.78,max_node+1.02],color="#4da83b",lw=2.4)
    ax.plot([0,.15],[max_node+.78,max_node+1.02],color="#4da83b",lw=2.4)

    ax.set_xlim(-2.45,2.45)
    ax.set_ylim(ground-.15,max_node+1.35)
    ax.axis("off")
    fig.tight_layout()
    return fig

def fig_to_png(fig):
    b=BytesIO(); fig.savefig(b,format="png",dpi=220,bbox_inches="tight"); b.seek(0); return b
def fig_to_pdf(fig):
    b=BytesIO(); fig.savefig(b,format="pdf",bbox_inches="tight"); b.seek(0); return b

st.markdown("""
<style>
:root{--navy:#062d57;--green:#078447;--border:#d8e3ec;--soft:#f8fbfd;}
.block-container{max-width:1550px;padding-top:.65rem;padding-bottom:1rem}
[data-testid="stSidebar"]{display:none}
h1,h2,h3,h4{color:var(--navy)}
.top-title{font-size:31px;font-weight:800;color:var(--navy);line-height:1}
.top-sub{font-size:15px;color:#315d8a;margin-top:6px}
.report-title{color:var(--green);font-weight:800;font-size:18px;margin-bottom:5px}
.panel{border:1px solid var(--border);border-radius:12px;background:#fff;padding:12px 14px}
.legend{border:1px solid var(--border);border-radius:12px;background:#fff;padding:12px 14px;margin-bottom:12px}
.metric-card{border:1px solid var(--border);border-radius:12px;background:linear-gradient(#fff,#f7fbfb);padding:12px;text-align:center;min-height:90px}
.metric-label{font-size:14px;color:#17395f}
.metric-value{font-size:27px;font-weight:800;color:#0b7a38;margin-top:4px}
.stTabs [data-baseweb="tab-list"]{gap:4px}
.stTabs [data-baseweb="tab"]{height:45px;padding:0 16px;color:var(--navy);font-weight:700}
.stTabs [aria-selected="true"]{color:var(--green)!important;border-bottom:3px solid var(--green)!important}
div[data-testid="stDataEditor"]{border:1px solid var(--border);border-radius:8px;overflow:hidden}
.stButton>button,.stDownloadButton>button{border-radius:8px!important;font-weight:700!important}
</style>
""", unsafe_allow_html=True)

logo=Path(__file__).with_name("agnvet_rural_logo.png")
hc1,hc2,hc3=st.columns([1.0,2.7,2.2],vertical_alignment="center")
with hc1:
    if logo.exists(): st.image(str(logo),width=190)
with hc2:
    st.markdown('<div class="top-title">Cotton Plant Mapper 🌱</div><div class="top-sub">Accurate mapping. Better decisions.</div>',unsafe_allow_html=True)
with hc3:
    top_pdf,top_png,top_clear=st.columns(3)
    pdf_slot=top_pdf.empty()
    png_slot=top_png.empty()
    clear_map=top_clear.button("🗑 Clear Map",use_container_width=True)

st.markdown('<div class="panel"><div class="report-title">▽ &nbsp; Report Details</div></div>',unsafe_allow_html=True)
r1,r2,r3,r4,r5,r6=st.columns([1.4,1.4,1.2,.85,.7,.7])
farm=r1.text_input("Farm",value="")
paddock=r2.text_input("Paddock Name",value="")
grower=r3.text_input("Grower",value="")
report_date=r4.date_input("Date")
min_node=r5.number_input("Lowest Node",1,50,1,1)
max_node=r6.number_input("Max Nodes",int(min_node),60,max(22,int(min_node)),1)

if "plant_matrix" not in st.session_state:
    st.session_state.plant_matrix=blank_matrix(int(min_node),int(max_node))
st.session_state.plant_matrix=normalize_matrix(st.session_state.plant_matrix,int(min_node),int(max_node))
if clear_map:
    st.session_state.plant_matrix=blank_matrix(int(min_node),int(max_node)); st.rerun()

left,centre,right=st.columns([1.22,1.52,.46],gap="medium")

show_labels=True; show_positions=True; show_ground=True

with left:
    t1,t2,t3=st.tabs(["📋 Data Entry","☷ Summary","⚙ Settings"])
    with t1:
        st.markdown("### Node Entry")
        a,b,c=st.columns(3)
        a.success("**V  Vegetative**")
        b.success("**R  Reproductive**")
        c.warning("**VL  Vegetative Lateral**")
        edited=st.data_editor(
            st.session_state.plant_matrix,use_container_width=True,hide_index=True,num_rows="fixed",height=520,
            column_config={
                "Node":st.column_config.NumberColumn("Node",disabled=True,width="small"),
                "Node Type":st.column_config.SelectboxColumn("Type",options=NODE_TYPES,required=True),
                "Position Count":st.column_config.NumberColumn("Positions",min_value=0,max_value=MAX_POSITIONS,step=1,width="small"),
                **{f"Position {p}":st.column_config.SelectboxColumn(f"Pos {p}",options=FRUIT_TYPES,required=True) for p in range(1,MAX_POSITIONS+1)},
                "Notes":st.column_config.TextColumn("Notes"),
            },
            column_order=["Node","Node Type","Position 1","Position 2","Position 3","Position Count","Position 4","Position 5","Position 6","Notes"],
            key="mapper_editor"
        )
        st.session_state.plant_matrix=normalize_matrix(edited,int(min_node),int(max_node))
        c1,c2,c3=st.columns(3)
        if c1.button("＋ Add Node",use_container_width=True):
            st.info("Increase **Max Nodes** above to add another node.")
        if c2.button("− Remove Last",use_container_width=True):
            st.info("Reduce **Max Nodes** above to remove the last node.")
        if c3.button("🪄 Auto Fill",use_container_width=True):
            df=st.session_state.plant_matrix.copy()
            for i,row in df.iterrows():
                if row["Node Type"]=="Reproductive" and row["Position Count"]>0 and row["Position 1"]=="-":
                    df.at[i,"Position 1"]="Square"
            st.session_state.plant_matrix=df; st.rerun()
    with t2:
        m=calculate_metrics(st.session_state.plant_matrix)
        st.metric("Total Nodes",m["total_nodes"])
        st.metric("Total Positions",m["total_positions"])
        st.metric("Held Positions",m["held_positions"])
        st.metric("Missing Fruit",m["missing_positions"])
        st.metric("Retention %",f'{m["retention"]:.1f}%' if m["retention"] is not None else "—")
    with t3:
        show_labels=st.toggle("Show Labels",True)
        show_positions=st.toggle("Show Positions",True)
        show_ground=st.toggle("Show Ground Line",True)
        st.toggle("Compact View",False)

with centre:
    st.markdown("<h3 style='text-align:center;margin:0 0 5px'>Cotton Plant Map</h3>",unsafe_allow_html=True)
    fig=make_figure(st.session_state.plant_matrix,int(min_node),int(max_node),farm,paddock,grower,
                    report_date.strftime("%d/%m/%Y"),show_labels,show_positions,show_ground)
    st.pyplot(fig,use_container_width=True)
    png=fig_to_png(fig); pdf=fig_to_pdf(fig)
    with pdf_slot.container():
        st.download_button("⇩ Export PDF",pdf,"cotton_plant_map.pdf","application/pdf",use_container_width=True)
    with png_slot.container():
        st.download_button("⇩ Export PNG",png,"cotton_plant_map.png","image/png",use_container_width=True)
    plt.close(fig)

with right:
    st.markdown('<div class="legend"><h4>Legend</h4>🌿 &nbsp; Boll<br><br>🌸 &nbsp; White Flower<br><br>✹ &nbsp; Square<br><br>◐ &nbsp; Cracked Boll<br><br>◌ &nbsp; Missing Fruit<br><br>○ &nbsp; Position (Empty)</div>',unsafe_allow_html=True)
    st.markdown('<div class="legend"><h4>Node Types</h4>🟢 &nbsp; <b>R</b> Reproductive<br><br>🔵 &nbsp; <b>V</b> Vegetative<br><br>🟠 &nbsp; <b>VL</b> Vegetative Lateral</div>',unsafe_allow_html=True)

m=calculate_metrics(st.session_state.plant_matrix)
st.write("")
cols=st.columns(5)
cards=[
    ("🌱 Total Nodes",m["total_nodes"]),
    ("○ Total Positions",m["total_positions"]),
    ("🌿 Held Positions",m["held_positions"]),
    ("◌ Missing Fruit",m["missing_positions"]),
    ("▥ Retention %",f'{m["retention"]:.1f}%' if m["retention"] is not None else "—"),
]
for col,(lab,val) in zip(cols,cards):
    col.markdown(f'<div class="metric-card"><div class="metric-label">{lab}</div><div class="metric-value">{val}</div></div>',unsafe_allow_html=True)
