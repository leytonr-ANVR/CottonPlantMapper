
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from io import BytesIO
from pathlib import Path
from datetime import date
import numpy as np
import base64

st.set_page_config(page_title="Cotton Plant Mapper", page_icon="🌱", layout="wide")

FRUIT_TYPES = ["-", "Boll", "White Flower", "Square", "Cracked Boll", "Missing Fruit"]
NODE_TYPES = ["Vegetative", "Reproductive", "Vegetative Lateral"]
MAX_POSITIONS = 6

FRUIT_DISPLAY = {
    "-": "—",
    "Boll": "🌿  Boll",
    "White Flower": "🌸  White Flower",
    "Square": "🟩  Square",
    "Cracked Boll": "◐  Cracked Boll",
    "Missing Fruit": "◌  Missing Fruit",
}
TYPE_DISPLAY = {
    "Vegetative": "🔵 V",
    "Reproductive": "🟢 R",
    "Vegetative Lateral": "🟠 VL",
}

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

def normalise(df, min_node, max_node):
    base = blank_matrix(min_node, max_node)
    if df is None or df.empty:
        return base
    df = df.copy()
    if "Node" not in df.columns:
        return base
    df["Node"] = pd.to_numeric(df["Node"], errors="coerce")
    df = df.dropna(subset=["Node"])
    df["Node"] = df["Node"].astype(int)

    for c in base.columns:
        if c not in df.columns:
            if c.startswith("Position "):
                df[c] = "-"
            elif c == "Node Type":
                df[c] = "Reproductive"
            elif c == "Position Count":
                df[c] = 3
            else:
                df[c] = ""

    df["Node Type"] = df["Node Type"].where(df["Node Type"].isin(NODE_TYPES), "Reproductive")
    df["Position Count"] = pd.to_numeric(df["Position Count"], errors="coerce").fillna(3).astype(int).clip(0, MAX_POSITIONS)
    for p in range(1, MAX_POSITIONS + 1):
        c = f"Position {p}"
        df[c] = df[c].where(df[c].isin(FRUIT_TYPES), "-")

    keep = list(base.columns)
    out = base.merge(df[keep], on="Node", how="left", suffixes=("", "_saved"))
    for c in keep[1:]:
        saved = f"{c}_saved"
        if saved in out.columns:
            out[c] = out[saved].fillna(out[c])

    out.loc[out["Node Type"] == "Vegetative", "Position Count"] = 0
    for p in range(1, MAX_POSITIONS + 1):
        out.loc[out["Position Count"] < p, f"Position {p}"] = "-"
    return out[keep]

def metrics(df):
    """Summary metrics. Vegetative nodes are excluded from fruit retention."""
    total_nodes = len(df)

    # Only reproductive-bearing node types count toward retention.
    retention_df = df[df["Node Type"] != "Vegetative"].copy()

    total_positions = 0
    held_positions = 0
    missing_positions = 0

    for _, row in retention_df.iterrows():
        count = int(row["Position Count"])
        total_positions += count

        for p in range(1, count + 1):
            fruit = row[f"Position {p}"]
            if fruit == "Missing Fruit":
                missing_positions += 1
            elif fruit != "-":
                held_positions += 1

    retention = (
        (held_positions / total_positions) * 100
        if total_positions > 0 else None
    )

    return {
        "total_nodes": total_nodes,
        "total_positions": total_positions,
        "held_positions": held_positions,
        "missing_positions": missing_positions,
        "retention": retention,
    }


def draw_square(ax, x, y, s=.12):
    ax.add_patch(Ellipse((x,y), s*.72, s*.84, facecolor="#5dad37", edgecolor="#347b31", lw=.8, zorder=9))
    for ang in [0,45,90,135,180,225,270,315]:
        a = np.deg2rad(ang)
        tip = (x + np.cos(a)*s*1.28, y + np.sin(a)*s*1.28)
        l = (x + np.cos(a+.20)*s*.30, y + np.sin(a+.20)*s*.30)
        r = (x + np.cos(a-.20)*s*.30, y + np.sin(a-.20)*s*.30)
        ax.add_patch(Polygon([l,tip,r], closed=True, facecolor="#248d38", edgecolor="#1f7130", lw=.5, zorder=10))

def draw_flower(ax, x, y, s=.15):
    petals = [(-.42,.10,18),(-.18,.40,62),(.18,.40,118),(.43,.10,162),(.25,-.18,215),(-.25,-.18,325)]
    for dx,dy,ang in petals:
        ax.add_patch(Ellipse((x+dx*s,y+dy*s), s*.95, s*.72, angle=ang,
                             facecolor="#fffdfd", edgecolor="#eadde5", lw=.75, zorder=9))
    ax.add_patch(Circle((x,y+.02*s), s*.22, facecolor="#f1cbd8", edgecolor="#d8a8b8", lw=.6, zorder=11))
    for ang in [205,240,275,310,345]:
        a=np.deg2rad(ang)
        tip=(x+np.cos(a)*s*1.14,y+np.sin(a)*s*1.03)
        l=(x+np.cos(a+.17)*s*.34,y+np.sin(a+.17)*s*.32)
        r=(x+np.cos(a-.17)*s*.34,y+np.sin(a-.17)*s*.32)
        ax.add_patch(Polygon([l,tip,r],closed=True,facecolor="#31943b",edgecolor="#26752f",lw=.5,zorder=8))

def _draw_interactive_image(ax, x, y, filename, fallback, s=.14, visual_zoom=.46):
    """Render custom fruit artwork at a consistent visible size."""
    symbol_path = Path(__file__).with_name(filename)
    if not symbol_path.exists():
        fallback(ax, x, y, s)
        return
    try:
        img = plt.imread(str(symbol_path))

        # Crop transparent/light outer padding so each artwork is scaled by the
        # actual fruit/flower rather than by the source image dimensions.
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            mask = alpha > 0.04
        else:
            rgb = img[:, :, :3]
            mask = np.min(rgb, axis=2) < 0.965

        ys, xs = np.where(mask)
        if len(xs) and len(ys):
            pad = 2
            x0, x1 = max(0, xs.min()-pad), min(img.shape[1], xs.max()+pad+1)
            y0, y1 = max(0, ys.min()-pad), min(img.shape[0], ys.max()+pad+1)
            img = img[y0:y1, x0:x1]

        artist = AnnotationBbox(
            OffsetImage(img, zoom=visual_zoom),
            (x, y),
            frameon=False,
            box_alignment=(0.5, 0.5),
            zorder=14,
            pad=0,
        )
        ax.add_artist(artist)
    except Exception:
        fallback(ax, x, y, s)


def draw_boll_image(ax, x, y, s=.14):
    _draw_interactive_image(ax, x, y, "boll_symbol.png", draw_boll, s, visual_zoom=0.6359)


def draw_white_flower_image(ax, x, y, s=.14):
    _draw_interactive_image(ax, x, y, "white_flower_symbol.png", draw_flower, s, visual_zoom=0.0970)


def draw_cracked_boll_image(ax, x, y, s=.14):
    _draw_interactive_image(ax, x, y, "cracked_boll_symbol.png", draw_cracked, s, visual_zoom=0.8007)


def draw_square_image(ax, x, y, s=.14):
    _draw_interactive_image(ax, x, y, "square_symbol.png", draw_square, s, visual_zoom=0.4600)


def draw_boll(ax, x, y, s=.14):
    for dx,ang,sc in [(-.25,-14,.90),(0,0,1.05),(.25,14,.90)]:
        ax.add_patch(Ellipse((x+dx*s,y+.08*s), s*.78*sc, s*1.12, angle=ang,
                             facecolor="#55a630", edgecolor="#2f762b", lw=.9, zorder=9))
    for ang in [42,72,108,138,218,270,322]:
        a=np.deg2rad(ang)
        tip=(x+np.cos(a)*s*1.32,y+np.sin(a)*s*1.36)
        l=(x+np.cos(a+.20)*s*.40,y+np.sin(a+.20)*s*.38)
        r=(x+np.cos(a-.20)*s*.40,y+np.sin(a-.20)*s*.38)
        ax.add_patch(Polygon([l,tip,r],closed=True,facecolor="#2f963a",edgecolor="#24752f",lw=.55,zorder=10))

def draw_cracked(ax, x, y, s=.14):
    for dx in [-.26,0,.26]:
        ax.add_patch(Ellipse((x+dx*s,y+.08*s),s*.56,s*.72,
                             facecolor="#fffdf8",edgecolor="#d9d7d0",lw=.7,zorder=11))
    for ang in [28,68,112,152,220,270,320]:
        a=np.deg2rad(ang)
        tip=(x+np.cos(a)*s*1.35,y+np.sin(a)*s*1.38)
        l=(x+np.cos(a+.20)*s*.38,y+np.sin(a+.20)*s*.36)
        r=(x+np.cos(a-.20)*s*.38,y+np.sin(a-.20)*s*.36)
        ax.add_patch(Polygon([l,tip,r],closed=True,facecolor="#35933b",edgecolor="#26742f",lw=.55,zorder=10))

def draw_missing(ax, x, y, s=.12):
    ax.add_patch(Circle((x,y),s*.72,fill=False,edgecolor="#9a3d1f",lw=1.6,linestyle=(0,(4,3)),zorder=9))

def draw_empty(ax, x, y, s=.11):
    ax.add_patch(Circle((x,y),s*.70,facecolor="white",edgecolor="#087a35",lw=1.5,zorder=9))

def draw_symbol(ax, x, y, fruit, s=.13):
    if fruit == "Boll": draw_boll(ax,x,y,s)
    elif fruit == "White Flower": draw_flower(ax,x,y,s)
    elif fruit == "Square": draw_square(ax,x,y,s)
    elif fruit == "Cracked Boll": draw_cracked(ax,x,y,s)
    elif fruit == "Missing Fruit": draw_missing(ax,x,y,s)

def draw_interactive_symbol(ax, x, y, fruit, s=.13):
    if fruit == "Boll":
        draw_boll_image(ax, x, y, s)
    elif fruit == "White Flower":
        draw_white_flower_image(ax, x, y, s)
    elif fruit == "Cracked Boll":
        draw_cracked_boll_image(ax, x, y, s)
    elif fruit == "Square":
        draw_square_image(ax, x, y, s)
    else:
        draw_symbol(ax, x, y, fruit, s)

def make_map(df, min_node, max_node, show_labels=True, show_positions=True, show_ground=True):
    # Plant-map layout styled to match the supplied reference:
    # slim upright main stem, alternating fruiting branches, compact node labels,
    # short branches where trailing positions are unused, and vegetative laterals
    # sweeping diagonally from the lower plant.
    node_span = max_node - min_node + 1
    fig_h = 7.0 if node_span <= 24 else 7.8
    fig, ax = plt.subplots(figsize=(8.6, fig_h))
    ax.set_facecolor('white')

    ground = min_node - 1.05
    stem_top = max_node + .72

    # Main stem and base.
    ax.plot([0,0],[ground,stem_top], color='#008b43', lw=4.2,
            solid_capstyle='round', zorder=2)
    if show_ground:
        ax.plot([-2.22,2.22],[ground,ground], color='#9c5a13', lw=4.8,
                solid_capstyle='round', zorder=1)
    ax.add_patch(Polygon([(-.20,ground),(0,ground+.30),(.20,ground)],
                         closed=True, facecolor='#008b43', edgecolor='#008b43', zorder=3))

    # Terminal.
    ax.plot([0,-.13],[stem_top-.02,stem_top+.23], color='#41a63b', lw=2.3)
    ax.plot([0,.13],[stem_top-.02,stem_top+.23], color='#41a63b', lw=2.3)

    for node in range(min_node, max_node + 1):
        row_df = df[df['Node'] == node]
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        ntype = row['Node Type']
        side = -1 if node % 2 else 1
        y = node
        count = int(row['Position Count'])

        # Show node number plus node type directly on the interactive map.
        type_short = {
            "Vegetative": "V",
            "Reproductive": "R",
            "Vegetative Lateral": "VL",
        }[ntype]
        type_colour = {
            "Vegetative": "#4d8ed8",
            "Reproductive": "#20b95a",
            "Vegetative Lateral": "#f07d18",
        }[ntype]

        label_x = -0.10 if side > 0 else 0.10
        label_ha = "right" if side > 0 else "left"

        # Node number.
        ax.text(
            label_x,
            y + .02,
            str(node),
            fontsize=8.6,
            fontweight="bold",
            color="#101010",
            ha=label_ha,
            va="center",
            zorder=20,
        )

        # Coloured R / V / VL badge immediately beside the node number.
        badge_x = label_x + (-.22 if side > 0 else .22)
        ax.text(
            badge_x,
            y + .02,
            type_short,
            fontsize=6.8 if type_short != "VL" else 6.1,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            zorder=21,
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor=type_colour,
                edgecolor=type_colour,
                linewidth=.5,
            ),
        )

        # Reproductive node junction is a small green/open position-style circle.
        if ntype == 'Reproductive':
            ax.add_patch(Circle((0,y), .048, facecolor='white', edgecolor='#008b43', lw=1.25, zorder=8))

        if ntype == 'Vegetative':
            # Short vegetative branch with a subtle leaf, kept compact like the lower nodes.
            branch_len = .72
            ex = side * branch_len
            ax.plot([0, side*.34, ex], [y, y+.02, y+.18], color='#008b43', lw=2.25,
                    solid_capstyle='round', zorder=3)
            ax.add_patch(Ellipse((ex + side*.12, y+.20), .25, .12,
                                 angle=18*side, facecolor='#5dab42', edgecolor='#3d8937', lw=.55, zorder=4))
            continue

        # Last active fruit position controls the branch length.
        effective = 0
        for p in range(1, count + 1):
            if row[f'Position {p}'] != '-':
                effective = p

        draw_count = max(effective, 1 if count > 0 else 0)

        if ntype == 'Vegetative Lateral':
            # Vegetative lateral = the long primary branch. Fruiting positions
            # sit on short secondary/subtending fruiting branches off this main
            # lateral rather than directly on the vegetative stem.
            branch_len = 1.18 + max(0, draw_count - 1) * .45
            ex = side * branch_len
            ey = y - .36 - max(0, draw_count - 2) * .045
            bend_x = side * .50
            bend_y = y - .13

            ax.plot(
                [0, bend_x, ex],
                [y, bend_y, ey],
                color='#008b43', lw=2.75,
                solid_capstyle='round', solid_joinstyle='round', zorder=3
            )

            coords = []
            if effective > 0:
                for p in range(1, effective + 1):
                    frac = p / (effective + 1)

                    # Anchor the secondary branch along the long vegetative lateral.
                    anchor_x = side * (.34 + (branch_len - .42) * frac)
                    anchor_y = y - .07 - (.25 * frac)

                    # Short subtending fruiting branch grows away from the main
                    # lateral. Alternate the rise slightly to keep positions clear.
                    sub_len = .27 + .025 * ((p - 1) % 2)
                    fruit_x = anchor_x + side * sub_len
                    fruit_y = anchor_y + (.24 if p % 2 else .18)

                    ax.plot(
                        [anchor_x, fruit_x],
                        [anchor_y, fruit_y],
                        color='#008b43', lw=2.05,
                        solid_capstyle='round', zorder=4
                    )

                    # Small junction marker where the fruiting branch leaves the
                    # vegetative lateral, matching the cotton-branch schematic.
                    ax.add_patch(Circle(
                        (anchor_x, anchor_y), .040,
                        facecolor='white', edgecolor='#008b43', lw=1.15, zorder=6
                    ))

                    coords.append((p, fruit_x, fruit_y))

        else:
            # Fruiting branches change angle/shape at each node.
            # This gives the plant the natural stepped/sweeping appearance in
            # the supplied reference instead of repeating one branch angle.
            branch_len = .72 + max(0, draw_count - 1) * .46

            # Repeat a gentle set of natural branch profiles up the plant.
            angle_profiles = [
                (-.28, -.18),   # sweeping downward
                (-.16, -.04),   # shallow downward
                (-.04,  .08),   # nearly level
                ( .08,  .20),   # gentle rise
                ( .16,  .30),   # stronger rise
            ]
            inner_dy, outer_dy = angle_profiles[(node - min_node) % len(angle_profiles)]

            # Lower branches are generally longer and more sweeping.
            height_frac = (node - min_node) / max(1, (max_node - min_node))
            branch_len *= (1.12 - .18 * height_frac)

            first_abs = min(.46, branch_len * .48)
            mid_abs = min(branch_len * .76, first_abs + .46)

            first_x = side * first_abs
            mid_x = side * mid_abs
            end_x = side * branch_len

            first_y = y + inner_dy * .32
            mid_y = y + inner_dy
            end_y = y + outer_dy

            if draw_count <= 1:
                # Even single-position branches get their own node-specific angle.
                ax.plot(
                    [0, end_x],
                    [y, end_y],
                    color='#008b43', lw=2.35,
                    solid_capstyle='round', zorder=3
                )
            else:
                # Multi-position branch uses two bends for a more natural cotton
                # fruiting-branch shape.
                ax.plot(
                    [0, first_x, mid_x, end_x],
                    [y, first_y, mid_y, end_y],
                    color='#008b43', lw=2.35,
                    solid_capstyle='round',
                    solid_joinstyle='round',
                    zorder=3
                )

            coords = []
            if effective > 0:
                for p in range(1, effective + 1):
                    frac = 1.0 if effective == 1 else (p - 1) / (effective - 1)
                    x_abs = .43 + max(0, branch_len - .43) * frac
                    x = side * x_abs

                    # Interpolate along the bent branch so fruit follows the
                    # changing branch angle instead of sitting on a straight line.
                    if x_abs <= first_abs or first_abs >= branch_len:
                        local = x_abs / max(first_abs, .001)
                        py = y + (first_y - y) * local
                    elif x_abs <= mid_abs or mid_abs >= branch_len:
                        local = (x_abs - first_abs) / max(mid_abs - first_abs, .001)
                        py = first_y + (mid_y - first_y) * local
                    else:
                        local = (x_abs - mid_abs) / max(branch_len - mid_abs, .001)
                        py = mid_y + (end_y - mid_y) * local

                    coords.append((p, x, py))

        # Fruit/position marks. Position circles are drawn behind fruit, like the reference.
        for p, x, py in coords:
            fruit = row[f'Position {p}']
            if show_positions:
                draw_empty(ax, x, py, .105)
            if fruit != '-':
                draw_interactive_symbol(ax, x, py, fruit, .145)
            if show_labels:
                ax.text(
                    x + (.07 if side > 0 else -.07),
                    py + .10,
                    f'{node}-{p}',
                    fontsize=6.0,
                    color='#56616b',
                    ha='left' if side > 0 else 'right',
                    va='bottom',
                    zorder=15,
                )

    # Keep the plant large and centred, with less empty space than previous versions.
    ax.set_xlim(-2.45, 2.45)
    ax.set_ylim(ground-.10, stem_top+.42)
    ax.axis('off')
    fig.tight_layout(pad=.05)
    return fig


def make_pdf_report(df, min_node, max_node, farm='', paddock='', grower='', report_date='', show_labels=True):
    """Clean portrait PDF styled like the supplied cotton_plant_map (7).pdf."""
    # A4 portrait report. Keep the plant dominant while adding a compact header.
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.set_facecolor('white')

    report_metrics = metrics(df)

    # Compact report title and details.
    # AGnVET Rural logo in the PDF header.
    pdf_logo = Path(__file__).with_name("agnvet_rural_logo.png")
    if pdf_logo.exists():
        try:
            logo_img = plt.imread(str(pdf_logo))
            logo_ax = fig.add_axes([0.055, 0.925, 0.205, 0.062])
            logo_ax.imshow(logo_img)
            logo_ax.axis("off")
        except Exception:
            pass

    fig.text(
        0.285, 0.973, "Cotton Plant Map",
        ha="left", va="top",
        fontsize=18, fontweight="bold", color="#062d57"
    )

    detail_left = []
    detail_right = []
    if farm:
        detail_left.append(f"Farm: {farm}")
    if paddock:
        detail_left.append(f"Paddock: {paddock}")
    if grower:
        detail_right.append(f"Grower: {grower}")
    if report_date:
        detail_right.append(f"Date: {report_date}")

    fig.text(
        0.285, 0.946,
        "   |   ".join(detail_left) if detail_left else "Farm: -   |   Paddock: -",
        ha="left", va="top",
        fontsize=9.5, color="#3f4f5f"
    )
    fig.text(
        0.285, 0.928,
        "   |   ".join(detail_right) if detail_right else "Grower: -   |   Date: -",
        ha="left", va="top",
        fontsize=9.5, color="#3f4f5f"
    )

    retention_text = (
        f"{report_metrics['retention']:.1f}%"
        if report_metrics["retention"] is not None else "-"
    )
    summary_text = (
        f"Total Nodes: {report_metrics['total_nodes']}     "
        f"Total Positions: {report_metrics['total_positions']}     "
        f"Held Positions: {report_metrics['held_positions']}     "
        f"Missing Fruit: {report_metrics['missing_positions']}     "
        f"Retention: {retention_text}"
    )

    fig.text(
        0.50, 0.895, summary_text,
        ha="center", va="top",
        fontsize=9.4, fontweight="bold", color="#162c45",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="#f7fafb",
            edgecolor="#c7d4d8",
            linewidth=0.8
        )
    )

    ground = min_node - 1.08
    stem_top = max_node + .72

    # Main stem / ground / base.
    ax.plot([0, 0], [ground, stem_top], color='#008b43', lw=4.5,
            solid_capstyle='round', zorder=2)
    ax.plot([-2.38, 2.38], [ground, ground], color='#9c5a13', lw=5.2,
            solid_capstyle='round', zorder=1)
    ax.add_patch(Polygon([(-.21, ground), (0, ground+.30), (.21, ground)],
                         closed=True, facecolor='#008b43', edgecolor='#008b43', zorder=3))

    # Terminal fork only - no large heading or report block.
    ax.plot([0, -.15], [stem_top-.02, stem_top+.23], color='#41a63b', lw=2.6)
    ax.plot([0, .15], [stem_top-.02, stem_top+.23], color='#41a63b', lw=2.6)

    for node in range(min_node, max_node + 1):
        row_df = df[df['Node'] == node]
        if row_df.empty:
            continue
        row = row_df.iloc[0]
        ntype = row['Node Type']
        side = -1 if node % 2 else 1
        y = node
        count = int(row['Position Count'])

        # Large clean node number close to the stem, as in the supplied PDF.
        ax.text(-.08 if side > 0 else .08, y, str(node),
                fontsize=10.8, fontweight='bold', color='#151515',
                ha='right' if side > 0 else 'left', va='center', zorder=20)

        if ntype == 'Vegetative':
            branch_len = .92
            end_x = side * branch_len
            ax.plot([0, side*.40, end_x], [y, y+.01, y+.20],
                    color='#008b43', lw=2.8, solid_capstyle='round', zorder=3)
            # One simple terminal leaf, matching the lower-node style in the example.
            ax.add_patch(Ellipse((end_x + side*.13, y+.22), .30, .13,
                                 angle=17*side, facecolor='#62ae45',
                                 edgecolor='#3d8937', lw=.6, zorder=4))
            continue

        # Last non-dash position determines branch length.
        effective = 0
        for p in range(1, count + 1):
            if row[f'Position {p}'] != '-':
                effective = p
        draw_count = max(effective, 1 if count else 0)

        if ntype == 'Vegetative Lateral':
            branch_len = 1.20 + max(0, draw_count - 1) * .46
            end_x = side * branch_len
            end_y = y - .36 - max(0, draw_count - 2) * .045
            bend_x = side * .51
            bend_y = y - .13
            ax.plot(
                [0, bend_x, end_x], [y, bend_y, end_y],
                color='#008b43', lw=2.9,
                solid_capstyle='round', solid_joinstyle='round', zorder=3
            )

            coords = []
            if effective > 0:
                for p in range(1, effective + 1):
                    frac = p / (effective + 1)
                    anchor_x = side * (.35 + (branch_len - .43) * frac)
                    anchor_y = y - .07 - .25 * frac
                    sub_len = .28 + .025 * ((p - 1) % 2)
                    fruit_x = anchor_x + side * sub_len
                    fruit_y = anchor_y + (.24 if p % 2 else .18)

                    ax.plot(
                        [anchor_x, fruit_x], [anchor_y, fruit_y],
                        color='#008b43', lw=2.15,
                        solid_capstyle='round', zorder=4
                    )
                    ax.add_patch(Circle(
                        (anchor_x, anchor_y), .040,
                        facecolor='white', edgecolor='#008b43', lw=1.15, zorder=6
                    ))
                    coords.append((p, fruit_x, fruit_y))
        else:
            # Reproductive branch: almost horizontal with a gentle outward rise.
            branch_len = .80 + max(0, draw_count - 1) * .52
            first_x = side * min(.52, branch_len)
            end_x = side * branch_len
            end_y = y + (.08 if draw_count <= 1 else .17)
            ax.plot([0, first_x, end_x], [y, y+.01, end_y],
                    color='#008b43', lw=2.9, solid_capstyle='round', zorder=3)

            coords = []
            if effective > 0:
                for p in range(1, effective + 1):
                    frac = 1.0 if effective == 1 else (p-1)/(effective-1)
                    x = side * (.50 + (branch_len - .50) * frac)
                    py = y + .025 + (end_y - y - .025) * frac
                    coords.append((p, x, py))

        for p, x, py in coords:
            fruit = row[f'Position {p}']
            # The supplied PDF has only the active fruit visible, not empty circles.
            if fruit != '-':
                draw_interactive_symbol(ax, x, py, fruit, .145)
            if show_labels:
                ax.text(x + (.07 if side > 0 else -.07), py + .10,
                        f'{node}-{p}', fontsize=7.0, color='#68717b',
                        ha='left' if side > 0 else 'right', va='bottom', zorder=15)

    # Tight portrait framing: plant fills the page vertically with clean white margins.
    ax.set_xlim(-2.55, 2.55)
    ax.set_ylim(ground-.14, stem_top+.38)
    ax.axis('off')
    fig.subplots_adjust(left=.035, right=.965, top=.865, bottom=.025)
    return fig

def legend_figure():
    """Legend using the exact same custom symbols as the interactive map."""
    fig, ax = plt.subplots(figsize=(2.35, 3.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    items = [
        ("Boll", "Boll"),
        ("White Flower", "White Flower"),
        ("Square", "Square"),
        ("Cracked Boll", "Cracked Boll"),
        ("Missing Fruit", "Missing Fruit"),
        ("Position (Empty)", "Position (Empty)"),
    ]

    ys = np.linspace(.88, .12, len(items))
    for (label, fruit), y in zip(items, ys):
        if fruit == "Position (Empty)":
            draw_empty(ax, .20, y, .055)
        elif fruit == "Missing Fruit":
            draw_missing(ax, .20, y, .060)
        else:
            # Uses the same custom image dispatcher and exact scaling as the map.
            draw_interactive_symbol(ax, .20, y, fruit, .145)

        ax.text(
            .34, y, label,
            fontsize=10,
            color="#13243a",
            fontweight="600",
            ha="left",
            va="center"
        )

    fig.tight_layout(pad=.15)
    return fig

def fig_bytes(fig, fmt):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=220 if fmt == "png" else None, bbox_inches="tight")
    buf.seek(0)
    return buf

def interactive_map(png_bytes, height=610):
    encoded = base64.b64encode(png_bytes.getvalue()).decode("ascii")
    html = f"""
    <html>
    <head>
    <style>
      html,body{{margin:0;padding:0;background:white;font-family:Arial,sans-serif;}}
      .wrap{{border:1px solid #d8e3ec;border-radius:12px;background:#fff;overflow:hidden;}}
      .bar{{height:46px;display:flex;align-items:center;justify-content:space-between;padding:0 14px;border-bottom:1px solid #e4ebf1;}}
      .title{{font-size:20px;font-weight:700;color:#062d57;margin-left:auto;margin-right:auto;}}
      .controls{{display:flex;gap:6px;position:absolute;right:12px;}}
      button{{width:36px;height:32px;border:1px solid #d5e0e9;border-radius:7px;background:#fff;color:#062d57;font-weight:700;cursor:pointer;}}
      button.fit{{width:42px;}}
      .viewport{{height:{height-48}px;overflow:hidden;display:flex;align-items:center;justify-content:center;background:white;position:relative;}}
      #mapimg{{max-width:96%;max-height:96%;object-fit:contain;transform-origin:center center;transition:transform .12s ease;user-select:none;}}
</style>
    </head>
    <body>
      <div class="wrap">
        <div class="bar">
          <div class="title">Cotton Plant Map</div>
          <div class="controls">
            <button onclick="zoomOut()" title="Zoom out">−</button>
            <button onclick="zoomIn()" title="Zoom in">＋</button>
            <button class="fit" onclick="fitMap()">Fit</button>
          </div>
        </div>
        <div class="viewport">
          <img id="mapimg" src="data:image/png;base64,{encoded}">
        </div>
      </div>
      <script>
        let scale = 1;
        const img = document.getElementById('mapimg');
        function apply(){{ img.style.transform = `scale(${{scale}})`; }}
        function zoomIn(){{ scale = Math.min(scale + 0.15, 3.0); apply(); }}
        function zoomOut(){{ scale = Math.max(scale - 0.15, 0.45); apply(); }}
        function fitMap(){{ scale = 1; apply(); }}
        img.addEventListener('wheel', function(e){{
          e.preventDefault();
          scale = Math.max(0.45, Math.min(3.0, scale + (e.deltaY < 0 ? 0.10 : -0.10)));
          apply();
        }}, {{passive:false}});
      </script>
    </body>
    </html>
    """
    components.html(html, height=height, scrolling=False)

# ---------- UI ----------
st.markdown("""
<style>
:root{--navy:#062d57;--green:#078447;--border:#d8e3ec;--orange:#ef6c21;}
.block-container{width:min(98vw,1500px)!important;max-width:1500px!important;padding:.45rem .55rem 1rem!important;margin:auto!important;}
[data-testid="stSidebar"]{display:none}
html,body,[class*="css"]{font-size:14px}
.brand-title{font-size:31px;font-weight:800;color:var(--navy);line-height:1.05}
.brand-sub{font-size:14px;color:#315d8a;margin-top:4px}
.report-wrap{border:1px solid var(--border);border-radius:12px;padding:8px 12px 12px;background:#fff;margin-top:4px}
.report-head{color:var(--green);font-weight:800;font-size:17px;margin:1px 0 6px}
.metric-card{border:1px solid var(--border);border-radius:11px;padding:10px 8px;text-align:center;background:linear-gradient(#fff,#f7fbfb);min-height:82px}
.metric-label{font-size:13px;color:#17395f}.metric-value{font-size:25px;font-weight:800;color:#0b7a38;margin-top:2px}
.type-strip{border-radius:8px;padding:8px 5px;text-align:center;font-weight:700;font-size:12px}
.type-v{border:1px solid #cfe0f7;background:#eef5ff;color:#1f65b5}
.type-r{border:1px solid #bfe3cb;background:#eefaf2;color:#19833e}
.type-vl{border:1px solid #ffd5c0;background:#fff5ef;color:#ef6c21}
.stTabs [data-baseweb="tab"]{font-weight:700;color:var(--navy);padding-left:10px;padding-right:10px}
.stTabs [aria-selected="true"]{color:var(--green)!important}
.stButton>button,.stDownloadButton>button{border-radius:8px!important;font-weight:700!important;min-height:38px;font-size:12px}
[data-baseweb="input"]{min-height:38px!important}
.node-label{padding-top:9px;text-align:center;font-weight:800;color:#17395f}
.legend-card{border:1px solid var(--border);border-radius:12px;background:#fff;padding:8px}
hr{margin:.4rem 0!important}

/* Compact Node Type dropdown: R / V / VL badge-style display */
[data-testid="stSelectbox"] [data-baseweb="select"] > div{
    font-weight:700;
}
</style>
""", unsafe_allow_html=True)

if "lowest_node" not in st.session_state:
    st.session_state.lowest_node = 1
if "max_node" not in st.session_state:
    st.session_state.max_node = 22
if "visible_position_columns" not in st.session_state:
    st.session_state.visible_position_columns = 3

logo = Path(__file__).with_name("agnvet_rural_logo.png")
h1,h2,h3 = st.columns([1.05,2.9,2.45], vertical_alignment="center")
with h1:
    if logo.exists():
        st.image(str(logo), width=190)
with h2:
    st.markdown('<div class="brand-title">Cotton Plant Mapper 🌱</div><div class="brand-sub">Accurate mapping. Better decisions.</div>', unsafe_allow_html=True)
with h3:
    e1,e2,e3,e4 = st.columns(4)
    pdf_slot = e1.empty()
    png_slot = e2.empty()
    clear_clicked = e3.button("🗑 Clear Map", use_container_width=True)
    save_slot = e4.empty()

st.markdown('<div class="report-wrap"><div class="report-head">▽ &nbsp; Report Details</div>', unsafe_allow_html=True)
r1,r2,r3,r4,r5,r6 = st.columns([1.5,1.5,1.15,.86,.72,.72])
farm = r1.text_input("Farm", key="farm")
paddock = r2.text_input("Paddock Name", key="paddock")
grower = r3.text_input("Grower", key="grower")
report_date = r4.date_input("Date", value=date.today(), key="report_date")
min_node = int(r5.number_input("Lowest Node", 1, 50, step=1, key="lowest_node"))
max_node = int(r6.number_input("Max Nodes", min_node, 60, step=1, key="max_node"))
st.markdown('</div>', unsafe_allow_html=True)

if "plant_matrix" not in st.session_state:
    st.session_state.plant_matrix = blank_matrix(min_node, max_node)
st.session_state.plant_matrix = normalise(st.session_state.plant_matrix, min_node, max_node)

if clear_clicked:
    st.session_state.plant_matrix = blank_matrix(min_node, max_node)
    st.session_state.visible_position_columns = 3
    for key in list(st.session_state.keys()):
        if key.startswith("type_") or key.startswith("pos_"):
            del st.session_state[key]
    st.rerun()

left, centre, right = st.columns([1.23,1.58,.48], gap="small")
show_labels = True
show_positions = True
show_ground = True

with left:
    tab_data, tab_summary, tab_settings = st.tabs(["▣  Data Entry","☷  Summary","⚙  Settings"])

    with tab_data:
        st.markdown("### Node Entry")
        tv,tr,tvl = st.columns(3)
        tv.markdown('<div class="type-strip type-v">V &nbsp; Vegetative</div>', unsafe_allow_html=True)
        tr.markdown('<div class="type-strip type-r">R &nbsp; Reproductive</div>', unsafe_allow_html=True)
        tvl.markdown('<div class="type-strip type-vl">VL &nbsp; Vegetative Lateral</div>', unsafe_allow_html=True)

        visible_positions = int(st.session_state.visible_position_columns)
        visible_positions = max(1, min(MAX_POSITIONS, visible_positions))

        # Dynamic header: Node, Type, then however many position columns are enabled.
        header_weights = [.38, .72] + [1.08] * visible_positions
        header = st.columns(header_weights, gap="small")
        header[0].markdown("<div style='text-align:center;font-size:11px;font-weight:700;color:#17395f'>Node</div>", unsafe_allow_html=True)
        header[1].markdown("<div style='text-align:center;font-size:11px;font-weight:700;color:#17395f'>Type</div>", unsafe_allow_html=True)
        for p in range(1, visible_positions + 1):
            header[p + 1].markdown(
                f"<div style='text-align:center;font-size:11px;font-weight:700;color:#17395f'>Pos {p}</div>",
                unsafe_allow_html=True
            )
        st.divider()

        df_edit = st.session_state.plant_matrix.copy()
        for idx,row in df_edit.sort_values("Node", ascending=False).iterrows():
            node = int(row["Node"])
            row_cols = st.columns(header_weights, gap="small")
            cnode = row_cols[0]
            ctype = row_cols[1]
            position_cols = row_cols[2:]

            cnode.markdown(f'<div class="node-label">{node}</div>', unsafe_allow_html=True)

            current_type = row["Node Type"]
            new_type = ctype.selectbox(
                f"Type {node}",
                NODE_TYPES,
                index=NODE_TYPES.index(current_type),
                format_func=lambda x: TYPE_DISPLAY[x],
                key=f"type_{node}",
                label_visibility="collapsed"
            )

            if new_type != current_type:
                df_edit.loc[df_edit["Node"] == node, "Node Type"] = new_type
                if new_type == "Vegetative":
                    df_edit.loc[df_edit["Node"] == node, "Position Count"] = 0
                    for p in range(1, MAX_POSITIONS + 1):
                        df_edit.loc[df_edit["Node"] == node, f"Position {p}"] = "-"
                else:
                    if int(row["Position Count"]) == 0:
                        df_edit.loc[df_edit["Node"] == node, "Position Count"] = visible_positions
                    if new_type == "Reproductive" and row["Position 1"] == "-":
                        df_edit.loc[df_edit["Node"] == node, "Position 1"] = "Square"

            for p, col in enumerate(position_cols, start=1):
                cur = df_edit.loc[df_edit["Node"] == node, f"Position {p}"].iloc[0]
                val = col.selectbox(
                    f"Node {node} Pos {p}",
                    FRUIT_TYPES,
                    index=FRUIT_TYPES.index(cur) if cur in FRUIT_TYPES else 0,
                    format_func=lambda x: FRUIT_DISPLAY[x],
                    key=f"pos_{node}_{p}",
                    label_visibility="collapsed"
                )
                df_edit.loc[df_edit["Node"] == node, f"Position {p}"] = val

            if new_type != "Vegetative":
                # Position Count follows the number of displayed position columns.
                df_edit.loc[df_edit["Node"] == node, "Position Count"] = visible_positions

        st.session_state.plant_matrix = normalise(df_edit, min_node, max_node)

        b1,b2,b3 = st.columns(3)
        if b1.button(
            f"＋ Add Position Column ({visible_positions}/{MAX_POSITIONS})",
            use_container_width=True,
            disabled=visible_positions >= MAX_POSITIONS
        ):
            st.session_state.visible_position_columns = min(MAX_POSITIONS, visible_positions + 1)
            # Increase Position Count for fruiting node types.
            df = st.session_state.plant_matrix.copy()
            df.loc[df["Node Type"] != "Vegetative", "Position Count"] = st.session_state.visible_position_columns
            st.session_state.plant_matrix = df
            st.rerun()

        if b2.button(
            f"− Remove Position Column",
            use_container_width=True,
            disabled=visible_positions <= 1
        ):
            removed_position = visible_positions
            st.session_state.visible_position_columns = max(1, visible_positions - 1)
            df = st.session_state.plant_matrix.copy()
            # Clear the removed column so it no longer affects the map or metrics.
            df[f"Position {removed_position}"] = "-"
            df.loc[df["Node Type"] != "Vegetative", "Position Count"] = st.session_state.visible_position_columns
            st.session_state.plant_matrix = df

            # Remove stale widget state for the hidden column.
            for node_id in df["Node"].tolist():
                key = f"pos_{int(node_id)}_{removed_position}"
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        if b3.button("🪄 Auto Fill", use_container_width=True):
            df = st.session_state.plant_matrix.copy()
            for i,r in df.iterrows():
                if r["Node Type"] == "Reproductive" and int(r["Position Count"]) > 0 and r["Position 1"] == "-":
                    df.at[i,"Position 1"] = "Square"
            st.session_state.plant_matrix = df
            st.rerun()

    with tab_summary:
        m = metrics(st.session_state.plant_matrix)
        st.metric("Total Nodes", m["total_nodes"])
        st.metric("Total Positions", m["total_positions"])
        st.metric("Held Positions", m["held_positions"])
        st.metric("Missing Fruit", m["missing_positions"])
        st.metric("Retention %", f'{m["retention"]:.1f}%' if m["retention"] is not None else "—")

    with tab_settings:
        show_labels = st.toggle("Show Labels", True)
        show_positions = st.toggle("Show Positions", True)
        show_ground = st.toggle("Show Ground Line", True)
        compact = st.toggle("Compact View", False)
        st.caption("Use the + / − / Fit controls on the map to zoom without changing the exported report.")

with centre:
    fig = make_map(st.session_state.plant_matrix, min_node, max_node, show_labels, show_positions, show_ground)
    # PNG export uses make_map(), so it contains the same custom Boll,
    # White Flower, Square and Cracked Boll artwork as the interactive map.
    png = fig_bytes(fig, "png")
    interactive_map(png, height=640 if compact else 720)
    plt.close(fig)

    # Dedicated portrait PDF export matching the supplied reference layout.
    pdf_fig = make_pdf_report(
        st.session_state.plant_matrix,
        min_node,
        max_node,
        farm=farm,
        paddock=paddock,
        grower=grower,
        report_date=report_date.strftime("%d/%m/%Y") if report_date else "",
        show_labels=True,
    )
    pdf = fig_bytes(pdf_fig, "pdf")
    plt.close(pdf_fig)

    with pdf_slot.container():
        st.download_button("⇩ Export PDF", data=pdf, file_name="cotton_plant_map.pdf", mime="application/pdf", use_container_width=True)
    with png_slot.container():
        st.download_button("⇩ Export PNG", data=png, file_name="cotton_plant_map.png", mime="image/png", use_container_width=True)

    csv_bytes = st.session_state.plant_matrix.to_csv(index=False).encode("utf-8")
    with save_slot.container():
        st.download_button("Save", data=csv_bytes, file_name="cotton_plant_map_data.csv", mime="text/csv", use_container_width=True)

with right:
    lf = legend_figure()
    st.pyplot(lf, use_container_width=True)
    plt.close(lf)

    st.markdown("""
    <div class="legend-card">
      <h4 style="margin:2px 0 10px;color:#062d57">Node Types</h4>
      <div style="margin:7px 0"><span style="background:#20b95a;color:white;border-radius:12px;padding:2px 8px;font-weight:700">R</span> &nbsp; Reproductive</div>
      <div style="margin:7px 0"><span style="background:#4d8ed8;color:white;border-radius:12px;padding:2px 8px;font-weight:700">V</span> &nbsp; Vegetative</div>
      <div style="margin:7px 0"><span style="background:#f07d18;color:white;border-radius:12px;padding:2px 8px;font-weight:700">VL</span> &nbsp; Vegetative Lateral</div>
    </div>
    """, unsafe_allow_html=True)

m = metrics(st.session_state.plant_matrix)
st.write("")
cards = st.columns(5)
for col,(label,value) in zip(cards,[
    ("🌱 Total Nodes",m["total_nodes"]),
    ("○ Total Positions",m["total_positions"]),
    ("🌿 Held Positions",m["held_positions"]),
    ("◌ Missing Fruit",m["missing_positions"]),
    ("▥ Retention %",f'{m["retention"]:.1f}%' if m["retention"] is not None else "—"),
]):
    col.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True
    )
