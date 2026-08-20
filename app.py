
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from io import BytesIO

st.set_page_config(page_title="Cotton Plant Mapper", page_icon="🌱", layout="wide")

FRUIT_TYPES = ["-", "Boll", "Square", "White Flower", "Missing Fruit"]
POSITIONS = [1, 2, 3]

def blank_matrix(min_node=5, max_node=22):
    return pd.DataFrame({
        "Node": list(range(min_node, max_node + 1)),
        "Position 1": ["-"] * (max_node - min_node + 1),
        "Position 2": ["-"] * (max_node - min_node + 1),
        "Position 3": ["-"] * (max_node - min_node + 1),
        "Notes": [""] * (max_node - min_node + 1),
    })

def normalize_matrix(df, min_node, max_node):
    base = blank_matrix(min_node, max_node)
    if df is None or df.empty:
        return base

    df = df.copy()
    if "Node" not in df.columns:
        return base

    df["Node"] = pd.to_numeric(df["Node"], errors="coerce")
    df = df.dropna(subset=["Node"])
    df["Node"] = df["Node"].astype(int)

    for col in ["Position 1", "Position 2", "Position 3"]:
        if col not in df.columns:
            df[col] = "-"
        df[col] = df[col].where(df[col].isin(FRUIT_TYPES), "-")

    if "Notes" not in df.columns:
        df["Notes"] = ""

    keep = df[["Node", "Position 1", "Position 2", "Position 3", "Notes"]]
    out = base.merge(keep, on="Node", how="left", suffixes=("", "_old"))

    for col in ["Position 1", "Position 2", "Position 3", "Notes"]:
        old = f"{col}_old"
        if old in out.columns:
            out[col] = out[old].fillna(out[col])

    return out[["Node", "Position 1", "Position 2", "Position 3", "Notes"]]

def matrix_to_long(df):
    rows = []
    for _, row in df.iterrows():
        node = int(row["Node"])
        for pos in POSITIONS:
            rows.append({
                "Node": node,
                "Position": pos,
                "Fruit": row[f"Position {pos}"]
            })
    return pd.DataFrame(rows)

def draw_symbol(ax, x, y, fruit, size=0.13):
    if fruit == "Boll":
        ax.add_patch(Circle((x, y), size, facecolor="black", edgecolor="black", lw=1.2, zorder=5))
    elif fruit == "Square":
        ax.scatter([x], [y], marker="P", s=90, c="black", zorder=5)
    elif fruit == "White Flower":
        ax.scatter([x], [y], marker="D", s=70, facecolors="white", edgecolors="black", linewidths=1.3, zorder=5)
    elif fruit == "Missing Fruit":
        ax.scatter([x], [y], marker="x", s=80, c="black", linewidths=1.6, zorder=5)

def make_figure(matrix_df, min_node, max_node, title, show_labels=True):
    df = matrix_to_long(matrix_df)
    fig_h = max(8, (max_node - min_node + 1) * 0.48)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))

    ax.plot([0, 0], [min_node - 0.8, max_node + 0.8], color="black", lw=5, solid_capstyle="round")

    for node in range(min_node, max_node + 1):
        side = -1 if node % 2 else 1
        y = node

        ax.plot([0, -0.32 * side], [y, y + 0.12], color="black", lw=1.1)

        xs = [0.45 * side, 0.88 * side, 1.27 * side]
        ys = [y + 0.02, y + 0.13, y + 0.25]
        ax.plot([0, xs[0], xs[1], xs[2]], [y, ys[0], ys[1], ys[2]], color="black", lw=1.2)

        for pos, x, py in zip(POSITIONS, xs, ys):
            row = df[(df["Node"] == node) & (df["Position"] == pos)]
            fruit = row.iloc[0]["Fruit"] if len(row) else "-"
            draw_symbol(ax, x, py, fruit)

            if show_labels:
                label_x = x + (0.12 if side > 0 else -0.12)
                ha = "left" if side > 0 else "right"
                ax.text(label_x, py + 0.10, f"{node}-{pos}", fontsize=7, ha=ha, va="bottom")

        ax.text(
            0.10 if side < 0 else -0.10,
            y,
            str(node),
            fontsize=8,
            ha="left" if side < 0 else "right",
            va="center",
        )

    ax.set_title(title, fontsize=15, pad=14)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(min_node - 1.1, max_node + 1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="black",
                   markeredgecolor="black", markersize=8, label="Boll"),
        plt.Line2D([0], [0], marker="P", color="black", linestyle="None",
                   markersize=8, label="Square"),
        plt.Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
                   markeredgecolor="black", markersize=7, label="White Flower"),
        plt.Line2D([0], [0], marker="x", color="black", linestyle="None",
                   markersize=8, label="Missing Fruit"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.035),
              ncol=4, frameon=False, fontsize=9)

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

st.title("🌱 Cotton Plant Mapper")
st.caption(
    "Enter each node across one row, with fruiting positions 1, 2 and 3 across the columns — like a traditional cotton plant mapping sheet."
)

with st.sidebar:
    st.header("Plant setup")
    min_node = st.number_input("Lowest node", min_value=1, max_value=50, value=5, step=1)
    max_node = st.number_input(
        "Highest node",
        min_value=int(min_node),
        max_value=60,
        value=max(22, int(min_node)),
        step=1,
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
        "Each **row is one node**. Use the three position columns to record the fruiting status. "
        "**Position 1** is closest to the main stem."
    )

    edited = st.data_editor(
        st.session_state.plant_matrix,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Node": st.column_config.NumberColumn(
                "Node",
                min_value=int(min_node),
                max_value=int(max_node),
                step=1,
                disabled=True,
                width="small",
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

    st.markdown("#### Entry key")
    key_df = pd.DataFrame({
        "Entry": ["-", "Boll", "Square", "White Flower", "Missing Fruit"],
        "Meaning": [
            "No fruit recorded",
            "Boll present",
            "Square / bud present",
            "White flower present",
            "Fruit missing / shed",
        ],
    })
    st.dataframe(key_df, use_container_width=True, hide_index=True)

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
        ["Boll", "Square", "White Flower", "Missing Fruit"], fill_value=0
    )

    cols = st.columns(4)
    for col, label in zip(cols, counts.index):
        col.metric(label, int(counts[label]))

    occupied = int(
        long_df["Fruit"].isin(["Boll", "Square", "White Flower"]).sum()
    )
    missing = int((long_df["Fruit"] == "Missing Fruit").sum())
    recorded = occupied + missing

    st.markdown("#### Fruiting summary")
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
st.caption(
    "Designed for fast plant mapping: scan down the nodes and record Positions 1, 2 and 3 across each row."
)
