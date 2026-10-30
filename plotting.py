import colorsys
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Patch, Rectangle, Wedge
from sklearn.neighbors import NearestNeighbors


MAJOR_COLORS = {
    'Epi': '#E41A1C',
    'TNKILC': '#377EB8',
    'Myeloid': '#4DAF4A',
    'Plasma': '#984EA3',
    'B': '#FF7F00',
    'Strom': '#A65628',
    'Mast': '#F781BF',
}


def stable_celltype_color(label):
    """Map a label to a color independently of label ordering or run subset."""
    label = str(label)
    if label in MAJOR_COLORS:
        return MAJOR_COLORS[label]
    digest = hashlib.sha256(label.encode('utf-8')).digest()
    hue = int.from_bytes(digest[:4], 'big') / 2**32
    saturation = 0.58 + (digest[4] / 255.0) * 0.24
    value = 0.72 + (digest[5] / 255.0) * 0.20
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return '#{:02X}{:02X}{:02X}'.format(*(round(channel * 255) for channel in rgb))


def resolve_celltype_colors(cell_types, colors_file=None):
    mapping = {}
    path = Path(colors_file) if colors_file else None
    if path and path.exists():
        existing = pd.read_csv(path, sep='\t')
        if {'cell_type', 'color'}.issubset(existing.columns):
            mapping.update(dict(zip(existing['cell_type'].astype(str), existing['color'].astype(str))))
    for label in sorted(map(str, cell_types)):
        mapping.setdefault(label, stable_celltype_color(label))
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sorted(mapping.items()), columns=['cell_type', 'color']).to_csv(
            path, sep='\t', index=False
        )
    return {str(label): mapping[str(label)] for label in cell_types}


def _automatic_radius(coordinates, scale=0.45):
    if len(coordinates) <= 1:
        return 1.0
    model = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(coordinates)
    distances, _ = model.kneighbors(coordinates)
    positive = distances[:, 1][distances[:, 1] > 0]
    return (float(np.median(positive)) if positive.size else 1.0) * float(scale)


def draw_spatial_pies(proportions, coordinates, color_map, save_path, plot_cfg, seed=0):
    """Plot predicted cell-type ratios at measured Cartesian/pixel locations."""
    coordinates = coordinates.loc[proportions.index]
    all_xy = coordinates.iloc[:, :2].to_numpy(dtype=float)
    max_spots = plot_cfg.get('max_spots')
    if max_spots is not None and len(proportions) > int(max_spots):
        rng = np.random.default_rng(int(seed))
        selected = np.sort(rng.choice(len(proportions), size=int(max_spots), replace=False))
        proportions = proportions.iloc[selected]
        coordinates = coordinates.iloc[selected]

    xy = coordinates.iloc[:, :2].to_numpy(dtype=float)
    radius = plot_cfg.get('radius')
    if radius is None or float(radius) <= 0:
        radius = _automatic_radius(all_xy, plot_cfg.get('radius_scale', 0.45))
    radius = float(radius)
    min_fraction = float(plot_cfg.get('min_fraction', 0.01))
    max_slices = int(plot_cfg.get('max_celltypes_per_spot', 8))
    cell_types = list(proportions.columns)
    values_all = proportions.to_numpy(dtype=float)
    values_all = np.clip(values_all, 0, None)

    patches = []
    facecolors = []
    spot_outlines = []
    other_used = False
    for (x, y), values in zip(xy, values_all):
        total = values.sum()
        if not np.isfinite(total) or total <= 0:
            continue
        spot_outlines.append(Circle((x, y), radius))
        fractions = values / total
        order = np.argsort(-fractions, kind='stable')
        keep = [idx for idx in order[:max_slices] if fractions[idx] >= min_fraction]
        remainder = max(0.0, 1.0 - float(fractions[keep].sum()))
        angle = 0.0
        for idx in keep:
            next_angle = angle + float(fractions[idx]) * 360.0
            patches.append(Wedge((x, y), radius, angle, next_angle))
            facecolors.append(color_map[cell_types[idx]])
            angle = next_angle
        if remainder > 1e-8:
            patches.append(Wedge((x, y), radius, angle, 360.0))
            facecolors.append('#BDBDBD')
            other_used = True

    figsize = plot_cfg.get('figsize', [14, 12])
    fig, ax = plt.subplots(figsize=tuple(figsize))
    rasterized = bool(plot_cfg.get('rasterized', True))
    collection = PatchCollection(
        patches,
        facecolor=facecolors,
        edgecolor=plot_cfg.get('edge_color', 'none'),
        linewidth=float(plot_cfg.get('edge_width', 0.0)),
        rasterized=rasterized,
    )
    ax.add_collection(collection)
    outline_collection = PatchCollection(
        spot_outlines,
        facecolor='none',
        edgecolor=plot_cfg.get('spot_boundary_color', 'black'),
        linewidth=float(plot_cfg.get('spot_boundary_width', 0.4)),
        rasterized=rasterized,
    )
    ax.add_collection(outline_collection)
    ax.set_xlim(xy[:, 0].min() - radius, xy[:, 0].max() + radius)
    ax.set_ylim(xy[:, 1].min() - radius, xy[:, 1].max() + radius)
    if bool(plot_cfg.get('invert_y', True)):
        ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    show_frame = bool(plot_cfg.get('show_frame', True))
    for spine in ax.spines.values():
        spine.set_visible(show_frame)
        spine.set_color(plot_cfg.get('frame_color', 'black'))
        spine.set_linewidth(float(plot_cfg.get('frame_width', 2.0)))
    ax.set_title(
        plot_cfg.get('title', 'Predicted spatial cell-type proportions'),
        fontsize=float(plot_cfg.get('title_fontsize', 22)),
        fontweight=plot_cfg.get('title_fontweight', 'bold'),
        pad=float(plot_cfg.get('title_pad', 16)),
    )

    if bool(plot_cfg.get('show_legend', True)):
        handles = [Patch(facecolor=color_map[label], label=label) for label in cell_types]
        if other_used:
            handles.append(Patch(facecolor='#BDBDBD', label='Other'))
        legend_columns = int(plot_cfg.get('legend_columns', 2 if len(handles) > 20 else 1))
        legend_inside = bool(plot_cfg.get('legend_inside', False))
        legend = ax.legend(
            handles=handles,
            title='Cell type',
            bbox_to_anchor=(0.02, 0.98) if legend_inside else (1.01, 1),
            loc='upper left',
            fontsize=float(plot_cfg.get('legend_fontsize', 7)),
            ncol=legend_columns,
            frameon=legend_inside,
            facecolor='white',
            edgecolor='black' if legend_inside else 'none',
            framealpha=float(plot_cfg.get('legend_framealpha', 0.9)),
        )
        legend.get_title().set_fontsize(float(plot_cfg.get('legend_title_fontsize', 18)))
        legend.get_title().set_fontweight('bold')
    fig.tight_layout()
    if bool(plot_cfg.get('show_outer_border', True)):
        fig.add_artist(
            Rectangle(
                (0.004, 0.004),
                0.992,
                0.992,
                transform=fig.transFigure,
                fill=False,
                edgecolor=plot_cfg.get('frame_color', 'black'),
                linewidth=float(plot_cfg.get('outer_border_width', 2.0)),
                clip_on=False,
                zorder=1000,
            )
        )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=int(plot_cfg.get('dpi', 300)))
    plt.close(fig)
    return len(proportions), radius
