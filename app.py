
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon
from io import BytesIO
import numpy as np

st.set_page_config(page_title="Cotton Plant Mapper", page_icon="🌿", layout="wide")

FRUIT_TYPES = ["-", "Boll", "Cracked Boll", "Square", "White Flower", "Missing Fruit"]
POSITIONS = [1, 2, 3]

def blank_matrix(min_node=5, max_node=22):
    return pd.DataFrame({
        "Node": list(range(min_node, max_node + 1)),
        "Node Type": ["Reproductive"] * (max_node - min_node + 1),
        "Position 1": ["-"] * (max_node - min_node + 1),
        "Position 2": ["-"] * (max_node - min_node + 1),
        "Position 3": ["-"] * (max_node - min_node + 1),
        "Notes": [""] * (max_node - min_node + 1),
    })

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
    df["Node Type"] = df["Node Type"].where(
        df["Node Type"].isin(["Vegetative", "Reproductive"]),
        "Reproductive"
    )

    for col in ["Position 1", "Position 2", "Position 3"]:
        if col not in df.columns:
            df[col] = "-"
        df[col] = df[col].where(df[col].isin(FRUIT_TYPES), "-")

    if "Notes" not in df.columns:
        df["Notes"] = ""

    out = base.merge(
        df[["Node", "Node Type", "Position 1", "Position 2", "Position 3", "Notes"]],
        on="Node", how="left", suffixes=("", "_saved")
    )
    for col in ["Node Type", "Position 1", "Position 2", "Position 3", "Notes"]:
        saved = f"{col}_saved"
        if saved in out.columns:
            out[col] = out[saved].fillna(out[col])
    return out[["Node", "Node Type", "Position 1", "Position 2", "Position 3", "Notes"]]

def matrix_to_long(df):
    rows = []
    for _, row in df.iterrows():
        node = int(row["Node"])
        for pos in POSITIONS:
            rows.append({
                "Node": node,
                "Node Type": row.get("Node Type", "Reproductive"),
                "Position": pos,
                "Fruit": row[f"Position {pos}"]
            })
    return pd.DataFrame(rows)

# ---------- Cotton-style fruit symbols ----------

def draw_square(ax, x, y, scale=0.12):
    # Green square/bud with three bracts.
    ax.add_patch(Circle((x, y), scale*0.62, facecolor="#6BAE3E",
                        edgecolor="#3F7F28", lw=1.0, zorder=8))
    for ang in [90, 210, 330]:
        a = np.deg2rad(ang)
        px = x + np.cos(a)*scale*0.58
        py = y + np.sin(a)*scale*0.58
        left = (x + np.cos(a+0.55)*scale*1.15,
                y + np.sin(a+0.55)*scale*1.15)
        right = (x + np.cos(a-0.55)*scale*1.15,
                 y + np.sin(a-0.55)*scale*1.15)
        tip = (x + np.cos(a)*scale*1.32,
               y + np.sin(a)*scale*1.32)
        ax.add_patch(Polygon([left, tip, right], closed=True,
                             facecolor="#62A93A", edgecolor="#3F7F28",
                             lw=0.7, zorder=7))

def draw_white_flower(ax, x, y, scale=0.16):
    # Five soft white petals with a light pink centre.
    for ang in np.linspace(0, 360, 5, endpoint=False):
        a = np.deg2rad(ang)
        px = x + np.cos(a)*scale*0.55
        py = y + np.sin(a)*scale*0.55
        ax.add_patch(Ellipse((px, py), scale*1.05, scale*0.72,
                             angle=ang, facecolor="white",
                             edgecolor="#E8E8E8", lw=0.9, zorder=8))
    ax.add_patch(Circle((x, y), scale*0.28, facecolor="#F7D6D8",
                        edgecolor="#D9A8AE", lw=0.7, zorder=9))
    # Green bracts beneath flower
    for ang in [205, 270, 335]:
        a = np.deg2rad(ang)
        tip = (x + np.cos(a)*scale*1.05,
               y + np.sin(a)*scale*1.05)
        l = (x + np.cos(a+0.35)*scale*0.35,
             y + np.sin(a+0.35)*scale*0.35)
        r = (x + np.cos(a-0.35)*scale*0.35,
             y + np.sin(a-0.35)*scale*0.35)
        ax.add_patch(Polygon([l, tip, r], closed=True,
                             facecolor="#5FA237", edgecolor="#417A2A",
                             lw=0.6, zorder=7))

def draw_boll(ax, x, y, scale=0.15):
    # Open cotton boll: brown base, white cotton locks.
    ax.plot([x, x], [y-scale*1.35, y-scale*0.65],
            color="#6A3D1F", lw=1.5, zorder=6)
    for dx, dy, s in [
        (-0.42, 0.08, 0.72),
        (0.00, 0.25, 0.82),
        (0.42, 0.08, 0.72),
        (-0.18, -0.20, 0.68),
        (0.20, -0.20, 0.68),
    ]:
        ax.add_patch(Circle((x+dx*scale, y+dy*scale),
                            scale*s, facecolor="white",
                            edgecolor="#D7D7D7", lw=0.8, zorder=9))
    for ang in [210, 270, 330]:
        a = np.deg2rad(ang)
        tip = (x + np.cos(a)*scale*1.25,
               y + np.sin(a)*scale*1.05)
        l = (x + np.cos(a+0.25)*scale*0.38,
             y + np.sin(a+0.25)*scale*0.30)
        r = (x + np.cos(a-0.25)*scale*0.38,
             y + np.sin(a-0.25)*scale*0.30)
        ax.add_patch(Polygon([l, tip, r], closed=True,
                             facecolor="#8B4D22", edgecolor="#653515",
                             lw=0.7, zorder=8))


def draw_cracked_boll(ax, x, y, scale=0.15):
    # Partially opened / cracked boll: green-brown boll with white cotton showing.
    ax.plot([x, x], [y-scale*1.25, y-scale*0.62],
            color="#6A3D1F", lw=1.5, zorder=6)

    # Outer boll segments
    for dx in [-0.34, 0.34]:
        ax.add_patch(Ellipse(
            (x + dx*scale, y),
            scale*0.78, scale*1.10,
            angle=-18 if dx < 0 else 18,
            facecolor="#8FAF45",
            edgecolor="#58752D",
            lw=0.9,
            zorder=8
        ))

    # Cotton visible through the cracked centre
    for dx, dy, s in [(-0.12, 0.08, 0.48), (0.12, 0.08, 0.48), (0, -0.12, 0.44)]:
        ax.add_patch(Circle(
            (x + dx*scale, y + dy*scale),
            scale*s,
            facecolor="white",
            edgecolor="#D7D7D7",
            lw=0.7,
            zorder=9
        ))

    # Brown split line
    ax.plot([x, x], [y-scale*0.50, y+scale*0.52],
            color="#70401F", lw=1.2, zorder=10)

    # Bracts
    for ang in [210, 270, 330]:
        a = np.deg2rad(ang)
        tip = (x + np.cos(a)*scale*1.20,
               y + np.sin(a)*scale*1.02)
        l = (x + np.cos(a+0.25)*scale*0.38,
             y + np.sin(a+0.25)*scale*0.30)
        r = (x + np.cos(a-0.25)*scale*0.38,
             y + np.sin(a-0.25)*scale*0.30)
        ax.add_patch(Polygon([l, tip, r], closed=True,
                             facecolor="#688D38", edgecolor="#466127",
                             lw=0.7, zorder=7))

def draw_missing(ax, x, y, scale=0.13):
    # Small fruiting scar/stub rather than a generic X.
    ax.plot([x-scale*0.55, x+scale*0.35],
            [y-scale*0.18, y+scale*0.10],
            color="#6B4B2B", lw=2.0, zorder=8)
    ax.add_patch(Circle((x+scale*0.42, y+scale*0.12),
                        scale*0.20, facecolor="#9B6A3B",
                        edgecolor="#5E3B20", lw=0.8, zorder=9))

def draw_symbol(ax, x, y, fruit, scale=0.13):
    if fruit == "Boll":
        draw_boll(ax, x, y, scale*1.05)
    elif fruit == "Cracked Boll":
        draw_cracked_boll(ax, x, y, scale*1.05)
    elif fruit == "Square":
        draw_square(ax, x, y, scale)
    elif fruit == "White Flower":
        draw_white_flower(ax, x, y, scale*1.05)
    elif fruit == "Missing Fruit":
        draw_missing(ax, x, y, scale)

def make_figure(matrix_df, min_node, max_node, title, show_labels=True):
    df = matrix_to_long(matrix_df)

    n_nodes = max_node - min_node + 1
    fig_h = max(9.5, n_nodes * 0.58)
    fig, ax = plt.subplots(figsize=(9.2, fig_h))
    ax.set_facecolor("#EAF5F8")

    # Ground
    ground_y = min_node - 1.15
    ax.fill_between([-2.8, 2.8], ground_y-0.35, ground_y,
                    color="#9A642C", zorder=0)

    # Main green stem with terminal.
    stem_x = 0
    ax.plot([stem_x, stem_x], [ground_y, max_node + 0.9],
            color="#3E8E45", lw=5.0, solid_capstyle="round", zorder=2)

    # Plant base
    ax.add_patch(Polygon([
        (-0.14, ground_y), (0.14, ground_y),
        (0.04, ground_y+0.28), (-0.04, ground_y+0.28)
    ], closed=True, facecolor="#3E8E45", edgecolor="#3E8E45", zorder=2))

    # Terminal shape
    ax.plot([0, -0.16], [max_node+0.9, max_node+1.16],
            color="#5BAE3E", lw=3.2, zorder=3)
    ax.plot([0, 0.16], [max_node+0.9, max_node+1.13],
            color="#5BAE3E", lw=3.2, zorder=3)
    ax.text(0, max_node+1.45, "Terminal", ha="center", va="center",
            fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="none", alpha=0.95))

    # Fruiting branches: more natural zig-zag shape, alternating sides.
    for node in range(min_node, max_node + 1):
        side = -1 if node % 2 else 1
        y = node

        node_row = matrix_df[matrix_df["Node"] == node]
        node_type = node_row.iloc[0]["Node Type"] if len(node_row) else "Reproductive"

        if node_type == "Vegetative":
            # Vegetative node: show a shorter vegetative branch without fruiting positions.
            branch_len = 0.95
            x1 = side * 0.42
            x2 = side * branch_len
            ax.plot([0, x1, x2], [y, y + 0.10, y + 0.28],
                    color="#5EAB3D", lw=3.0, solid_capstyle="round", zorder=3)

            # simple leaf shape
            leaf_x = x2 + side * 0.18
            leaf_y = y + 0.34
            ax.add_patch(Ellipse(
                (leaf_x, leaf_y), 0.34, 0.16,
                angle=25 if side > 0 else -25,
                facecolor="#6EAF42", edgecolor="#4E8A31",
                lw=0.8, zorder=4
            ))
            coords = []
        else:
            # Reproductive node: draw fruiting branch with three positions.
            branch_len = 1.45 + (0.42 if node < min_node + n_nodes*0.45 else 0.10)
            x0 = 0.0
            x1 = side * 0.52
            x2 = side * 1.05
            x3 = side * branch_len
            y1 = y + 0.03
            y2 = y + 0.20
            y3 = y + 0.34
            ax.plot([x0, x1, x2, x3], [y, y1, y2, y3],
                    color="#4F9C37", lw=3.0, solid_capstyle="round", zorder=3)

            coords = [
                (side * 0.48, y + 0.04),
                (side * 0.98, y + 0.18),
                (side * (branch_len - 0.05), y + 0.33),
            ]

        # node dots like the supplied reference
        node_colour = "#2E6E35" if node_type == "Vegetative" else "black"
        ax.add_patch(Circle((0, y), 0.095, facecolor=node_colour,
                            edgecolor="black", zorder=10))

        if node_type == "Reproductive":
            for pos, (x, py) in zip(POSITIONS, coords):
                row = df[(df["Node"] == node) & (df["Position"] == pos)]
                fruit = row.iloc[0]["Fruit"] if len(row) else "-"
                if fruit != "-":
                    draw_symbol(ax, x, py, fruit, scale=0.14)

                if show_labels:
                    off = 0.10 if side > 0 else -0.10
                    ax.text(x + off, py + 0.15, f"{node}-{pos}",
                            fontsize=7, color="#333333",
                            ha="left" if side > 0 else "right",
                            va="bottom", zorder=11)

        # Node number shown beside the stem
        type_mark = "V" if node_type == "Vegetative" else "R"
        ax.text(-0.18 if side > 0 else 0.18, y, f"{node} {type_mark}",
                fontsize=8, fontweight="bold", color="#1F1F1F",
                ha="right" if side > 0 else "left", va="center", zorder=11)

    ax.set_title(title, fontsize=17, fontweight="bold", pad=16)
    ax.set_xlim(-2.55, 2.55)
    ax.set_ylim(ground_y-0.20, max_node + 1.85)
    ax.axis("off")

    # Custom legend with the same cotton-style symbols
    lx = -2.25
    ly = ground_y + 0.32
    spacing = 0.72
    legend_items = ["Square", "White Flower", "Cracked Boll", "Boll", "Missing Fruit"]
    for i, fruit in enumerate(legend_items):
        yy = ly + i * spacing
        draw_symbol(ax, lx, yy, fruit, scale=0.13)
        ax.text(lx + 0.28, yy, fruit, ha="left", va="center",
                fontsize=9, color="#222222")

    fig.tight_layout()
    return fig

def fig_to_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    buf.seek(0)
    return buf

def fig_to_pdf(fig):
    buf = BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    buf.seek(0)
    return buf

st.title("🌿 Cotton Plant Mapper")
st.caption(
    "Map cotton fruiting by node and position, then generate a plant-style diagram using cotton squares, flowers, bolls and fruiting scars."
)

with st.sidebar:
    st.header("Plant setup")
    min_node = st.number_input("Lowest node", min_value=1, max_value=50, value=5, step=1)
    max_node = st.number_input(
        "Highest node",
        min_value=int(min_node),
        max_value=60,
        value=max(22, int(min_node)),
        step=1
    )
    plant_name = st.text_input("Plant / sample name", value="Cotton Plant Map")
    show_labels = st.checkbox("Show node-position labels", value=True)

if "plant_matrix" not in st.session_state:
    st.session_state.plant_matrix = blank_matrix(int(min_node), int(max_node))

st.session_state.plant_matrix = normalize_matrix(
    st.session_state.plant_matrix, int(min_node), int(max_node)
)

tab1, tab2, tab3 = st.tabs(["Data Entry", "Plant Map", "Summary"])

with tab1:
    st.subheader("Node × Position Entry")
    st.write(
        "Each row is one node. Set the node as **Vegetative** or **Reproductive**. "
        "Reproductive nodes use Positions **1, 2 and 3** across the row, with Position 1 closest to the main stem."
    )

    edited = st.data_editor(
        st.session_state.plant_matrix,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Node": st.column_config.NumberColumn(
                "Node", disabled=True, width="small"
            ),
            "Node Type": st.column_config.SelectboxColumn(
                "Node Type",
                options=["Vegetative", "Reproductive"],
                required=True,
                width="medium"
            ),
            "Position 1": st.column_config.SelectboxColumn(
                "1", options=FRUIT_TYPES, required=True, width="small"
            ),
            "Position 2": st.column_config.SelectboxColumn(
                "2", options=FRUIT_TYPES, required=True, width="small"
            ),
            "Position 3": st.column_config.SelectboxColumn(
                "3", options=FRUIT_TYPES, required=True, width="small"
            ),
            "Notes": st.column_config.TextColumn("Notes", width="medium"),
        },
        key="plant_matrix_editor",
    )

    st.session_state.plant_matrix = normalize_matrix(
        edited, int(min_node), int(max_node)
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear all fruit", use_container_width=True):
            st.session_state.plant_matrix = blank_matrix(int(min_node), int(max_node))
            st.rerun()
    with c2:
        st.download_button(
            "Download data as CSV",
            data=st.session_state.plant_matrix.to_csv(index=False).encode("utf-8"),
            file_name="cotton_plant_map_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    uploaded = st.file_uploader("Load a saved CSV", type=["csv"])
    if uploaded is not None:
        try:
            loaded = pd.read_csv(uploaded)
            st.session_state.plant_matrix = normalize_matrix(
                loaded, int(min_node), int(max_node)
            )
            st.success("CSV loaded.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

with tab2:
    fig = make_figure(
        st.session_state.plant_matrix,
        int(min_node),
        int(max_node),
        plant_name,
        show_labels,
    )
    st.pyplot(fig, use_container_width=False)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download map as PNG",
            data=fig_to_png(fig),
            file_name="cotton_plant_map.png",
            mime="image/png",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download map as PDF",
            data=fig_to_pdf(fig),
            file_name="cotton_plant_map.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    plt.close(fig)

with tab3:
    long_df = matrix_to_long(st.session_state.plant_matrix)
    active = long_df[long_df["Fruit"] != "-"].copy()
    counts = active["Fruit"].value_counts().reindex(
        ["Boll", "Cracked Boll", "Square", "White Flower", "Missing Fruit"], fill_value=0
    )

    cols = st.columns(5)
    for col, label in zip(cols, counts.index):
        col.metric(label, int(counts[label]))

    occupied = int(long_df["Fruit"].isin(["Boll", "Cracked Boll", "Square", "White Flower"]).sum())
    missing = int((long_df["Fruit"] == "Missing Fruit").sum())
    recorded = occupied + missing


    st.markdown("#### Node type summary")
    node_counts = st.session_state.plant_matrix["Node Type"].value_counts().reindex(
        ["Vegetative", "Reproductive"], fill_value=0
    )
    nc1, nc2 = st.columns(2)
    nc1.metric("Vegetative Nodes", int(node_counts["Vegetative"]))
    nc2.metric("Reproductive Nodes", int(node_counts["Reproductive"]))

    summary_df = pd.DataFrame({
        "Metric": [
            "Recorded fruiting sites",
            "Occupied fruiting sites",
            "Missing fruit",
            "Retention of recorded sites",
        ],
        "Value": [
            recorded,
            occupied,
            missing,
            f"{(occupied / recorded * 100):.1f}%" if recorded else "—",
        ],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.divider()
st.caption("Plant map styling updated to resemble a real cotton plant with natural green branches and cotton-specific fruit symbols.")
