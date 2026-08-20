
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from io import BytesIO

st.set_page_config(page_title="Cotton Plant Mapper", page_icon="🌱", layout="wide")

FRUIT_TYPES = ["None", "Boll", "Square", "White Flower", "Missing Fruit"]
POSITIONS = [1, 2, 3]

def blank_data(min_node=5, max_node=22):
    rows = []
    for node in range(min_node, max_node + 1):
        for pos in POSITIONS:
            rows.append({
                "Node": node,
                "Position": pos,
                "Fruit": "None",
                "Notes": ""
            })
    return pd.DataFrame(rows)

def normalize_df(df):
    df = df.copy()
    if "Node" not in df.columns:
        df["Node"] = 5
    if "Position" not in df.columns:
        df["Position"] = 1
    if "Fruit" not in df.columns:
        df["Fruit"] = "None"
    if "Notes" not in df.columns:
        df["Notes"] = ""

    df["Node"] = pd.to_numeric(df["Node"], errors="coerce").fillna(5).astype(int)
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce").fillna(1).astype(int)
    df["Position"] = df["Position"].clip(1, 3)
    df["Fruit"] = df["Fruit"].where(df["Fruit"].isin(FRUIT_TYPES), "None")
    return df.sort_values(["Node", "Position"]).reset_index(drop=True)

def draw_symbol(ax, x, y, fruit, size=0.13):
    if fruit == "Boll":
        ax.add_patch(Circle((x, y), size, facecolor="black", edgecolor="black", lw=1.2, zorder=5))
    elif fruit == "Square":
        ax.scatter([x], [y], marker="P", s=90, c="black", zorder=5)
    elif fruit == "White Flower":
        ax.scatter([x], [y], marker="D", s=70, facecolors="white", edgecolors="black", linewidths=1.3, zorder=5)
    elif fruit == "Missing Fruit":
        ax.scatter([x], [y], marker="x", s=80, c="black", linewidths=1.6, zorder=5)

def make_figure(df, min_node, max_node, title, show_labels=True):
    df = normalize_df(df)
    fig_h = max(8, (max_node - min_node + 1) * 0.48)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))

    # Main stem
    ax.plot([0, 0], [min_node - 0.8, max_node + 0.8], color="black", lw=5, solid_capstyle="round")

    # Draw alternating branches and position points.
    for node in range(min_node, max_node + 1):
        side = -1 if node % 2 else 1
        y = node
        # Short opposite leaf/petiole hint
        ax.plot([0, -0.32 * side], [y, y + 0.12], color="black", lw=1.1)

        # Fruiting branch with 3 positions
        xs = [0.45 * side, 0.88 * side, 1.27 * side]
        ys = [y + 0.02, y + 0.13, y + 0.25]
        ax.plot([0, xs[0], xs[1], xs[2]], [y, ys[0], ys[1], ys[2]], color="black", lw=1.2)

        for pos, x, py in zip(POSITIONS, xs, ys):
            row = df[(df["Node"] == node) & (df["Position"] == pos)]
            fruit = row.iloc[0]["Fruit"] if len(row) else "None"
            draw_symbol(ax, x, py, fruit)

            if show_labels:
                # Labels slightly offset away from branch
                label_x = x + (0.12 if side > 0 else -0.12)
                ha = "left" if side > 0 else "right"
                ax.text(label_x, py + 0.10, f"{node}-{pos}", fontsize=7, ha=ha, va="bottom")

        # Node number next to stem
        ax.text(0.10 if side < 0 else -0.10, y, str(node), fontsize=8,
                ha="left" if side < 0 else "right", va="center")

    ax.set_title(title, fontsize=15, pad=14)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(min_node - 1.1, max_node + 1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor='black',
                   markeredgecolor='black', markersize=8, label='Boll'),
        plt.Line2D([0], [0], marker='P', color='black', linestyle='None',
                   markersize=8, label='Square'),
        plt.Line2D([0], [0], marker='D', color='none', markerfacecolor='white',
                   markeredgecolor='black', markersize=7, label='White Flower'),
        plt.Line2D([0], [0], marker='x', color='black', linestyle='None',
                   markersize=8, label='Missing Fruit'),
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
st.caption("Enter fruiting data by node and position to generate a cotton plant map for bolls, squares, white flowers and missing fruit.")

with st.sidebar:
    st.header("Plant setup")
    min_node = st.number_input("Lowest node", min_value=1, max_value=50, value=5, step=1)
    max_node = st.number_input("Highest node", min_value=int(min_node), max_value=60, value=max(22, int(min_node)), step=1)
    plant_name = st.text_input("Plant / sample name", value="Cotton Plant Map")
    show_labels = st.checkbox("Show node-position labels", value=True)

    st.divider()
    st.subheader("Quick entry")
    q_node = st.number_input("Node", min_value=int(min_node), max_value=int(max_node), value=int(min_node))
    q_pos = st.selectbox("Position", POSITIONS, index=0)
    q_fruit = st.selectbox("Fruit", FRUIT_TYPES[1:] + ["None"], index=0)

if "plant_df" not in st.session_state:
    st.session_state.plant_df = blank_data(int(min_node), int(max_node))

# Reset data if node range changes substantially while preserving overlaps
current = normalize_df(st.session_state.plant_df)
wanted = blank_data(int(min_node), int(max_node))
merged = wanted.merge(
    current[["Node", "Position", "Fruit", "Notes"]],
    on=["Node", "Position"],
    how="left",
    suffixes=("", "_old")
)
merged["Fruit"] = merged["Fruit_old"].fillna(merged["Fruit"])
merged["Notes"] = merged["Notes_old"].fillna(merged["Notes"])
st.session_state.plant_df = merged[["Node", "Position", "Fruit", "Notes"]]

if st.sidebar.button("Add / update entry", use_container_width=True):
    mask = (st.session_state.plant_df["Node"] == int(q_node)) & (st.session_state.plant_df["Position"] == int(q_pos))
    st.session_state.plant_df.loc[mask, "Fruit"] = q_fruit
    st.rerun()

if st.sidebar.button("Clear all fruit", use_container_width=True):
    st.session_state.plant_df["Fruit"] = "None"
    st.session_state.plant_df["Notes"] = ""
    st.rerun()

tab1, tab2, tab3 = st.tabs(["Plant Map", "Data Entry", "Summary"])

with tab2:
    st.subheader("Enter fruiting data")
    st.write("Each row represents a fruiting position. Choose the status in the **Fruit** column. Position 1 is closest to the main stem.")
    edited = st.data_editor(
        st.session_state.plant_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Node": st.column_config.NumberColumn("Node", min_value=int(min_node), max_value=int(max_node), step=1, disabled=True),
            "Position": st.column_config.NumberColumn("Position", min_value=1, max_value=3, step=1, disabled=True),
            "Fruit": st.column_config.SelectboxColumn("Fruit", options=FRUIT_TYPES, required=True),
            "Notes": st.column_config.TextColumn("Notes")
        },
        key="plant_editor"
    )
    st.session_state.plant_df = normalize_df(edited)

    st.download_button(
        "Download data as CSV",
        data=st.session_state.plant_df.to_csv(index=False).encode("utf-8"),
        file_name="cotton_plant_map_data.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("Or load a saved CSV", type=["csv"])
    if uploaded is not None:
        try:
            loaded = pd.read_csv(uploaded)
            loaded = normalize_df(loaded)
            base = blank_data(int(min_node), int(max_node))
            keycols = ["Node", "Position"]
            combined = base.merge(loaded[keycols + ["Fruit", "Notes"]], on=keycols, how="left", suffixes=("", "_new"))
            combined["Fruit"] = combined["Fruit_new"].fillna("None")
            combined["Notes"] = combined["Notes_new"].fillna("")
            st.session_state.plant_df = combined[["Node", "Position", "Fruit", "Notes"]]
            st.success("CSV loaded. The map has been updated.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

with tab1:
    fig = make_figure(st.session_state.plant_df, int(min_node), int(max_node), plant_name, show_labels)
    st.pyplot(fig, use_container_width=False)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download map as PNG",
            data=fig_to_png(fig),
            file_name="cotton_plant_map.png",
            mime="image/png",
            use_container_width=True
        )
    with c2:
        st.download_button(
            "Download map as PDF",
            data=fig_to_pdf(fig),
            file_name="cotton_plant_map.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    plt.close(fig)

with tab3:
    df = normalize_df(st.session_state.plant_df)
    active = df[df["Fruit"] != "None"].copy()
    counts = active["Fruit"].value_counts().reindex(["Boll", "Square", "White Flower", "Missing Fruit"], fill_value=0)

    cols = st.columns(4)
    for col, label in zip(cols, counts.index):
        col.metric(label, int(counts[label]))

    total_sites = (int(max_node) - int(min_node) + 1) * 3
    retained = int((df["Fruit"] != "Missing Fruit").sum() - (df["Fruit"] == "None").sum())
    missing = int((df["Fruit"] == "Missing Fruit").sum())
    occupied = int((df["Fruit"].isin(["Boll", "Square", "White Flower"])).sum())

    st.markdown("#### Fruiting summary")
    summary_df = pd.DataFrame({
        "Metric": ["Total mapped positions", "Occupied fruiting sites", "Missing fruit", "Retention of recorded sites"],
        "Value": [
            total_sites,
            occupied,
            missing,
            f"{(occupied / (occupied + missing) * 100):.1f}%" if (occupied + missing) else "—"
        ]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if not active.empty:
        st.markdown("#### Recorded positions")
        st.dataframe(active, use_container_width=True, hide_index=True)
    else:
        st.info("No fruiting data entered yet.")

st.divider()
st.caption("Tip: Use the Data Entry tab for full editing, or Quick Entry in the sidebar when mapping a plant in the field.")
