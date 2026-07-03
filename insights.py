"""
Step 13 — Coaching Insights Generator
Reads tracking-fixed.csv and generates a match report + charts.
Usage: python insights.py --csv_path data/tracking-fixed.csv --output_dir data/insights
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PITCH_LENGTH_METERS = 105.0
PITCH_WIDTH_METERS = 68.0
# 38km/h is close to the fastest recorded football sprint speeds
# (e.g. Mbappe ~37.8km/h). 45km/h was too high a ceiling and let
# obvious glitch readings pass through as "valid" data.
MAX_REALISTIC_SPEED = 38.0  # km/h cap
FPS = 25.0

# Max physically realistic distance a player can move in one frame.
# At 25fps, even a 38km/h sprint = 38/3.6/25 = ~0.42m per frame.
# Anything beyond this is a homography/tracking glitch, not real movement,
# and gets discarded (not summed) when computing total distance covered.
MAX_STEP_DIST_M = 0.45

TEAM_COLORS = {0: '#FF1493', 1: '#00BFFF', 3: '#FFD700'}
TEAM_NAMES = {0: 'Team 0', 1: 'Team 1', 3: 'Referees'}


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Discard (don't clip!) unrealistic speed readings. Clipping a 500km/h
    # glitch down to 45km/h still injects a fake data point into every
    # average/max calculation downstream. Setting it to 0 excludes it
    # from the >0 filters used throughout this script instead.
    bad_speed = df['speed_kmh'] > MAX_REALISTIC_SPEED
    if bad_speed.sum() > 0:
        print(
            f"[Cleaning] Discarded {bad_speed.sum()} unrealistic speed "
            f"reading(s) (>{MAX_REALISTIC_SPEED}km/h)"
        )
    df.loc[bad_speed, 'speed_kmh'] = 0.0
    df = filter_ghost_tracks(df)
    return df


def filter_ghost_tracks(df: pd.DataFrame, min_frames: int = 100) -> pd.DataFrame:
    """
    Removes tracker_ids that only appear for a small number of frames.
    These are almost always ID switches (ByteTrack losing and re-acquiring
    a player) or false detections, not real distinct players, and they
    pollute "top distance" / "top speed" rankings if left in.
    """
    frame_counts = df.groupby('tracker_id')['frame'].transform('count')
    ghost_count = (frame_counts < min_frames).sum()
    if ghost_count > 0:
        ghost_ids = sorted(df.loc[frame_counts < min_frames, 'tracker_id'].unique())
        print(
            f"[Cleaning] Dropped {len(ghost_ids)} ghost tracker_id(s) "
            f"with <{min_frames} frames: {ghost_ids}"
        )
    return df[frame_counts >= min_frames].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------
def calc_possession(df: pd.DataFrame) -> dict:
    players = df[df['role'] == 'player']
    counts = players.groupby('team_id')['frame'].count()
    total = counts.sum()
    return {tid: round(count / total * 100, 1) for tid, count in counts.items()}


def calc_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['tracker_id', 'frame'])
    df['prev_frame'] = df.groupby('tracker_id')['frame'].shift(1)
    df['dx'] = df.groupby('tracker_id')['field_x_m'].diff()
    df['dy'] = df.groupby('tracker_id')['field_y_m'].diff()
    df['frame_gap'] = df['frame'] - df['prev_frame']
    df['raw_step_dist'] = np.where(
        df['frame_gap'] == 1,
        np.hypot(df['dx'], df['dy']),
        0.0
    )
    # Discard physically impossible single-frame jumps (homography/tracking
    # glitches), instead of summing them into the player's total distance.
    # Without this, a handful of bad frames can inflate distance by 50%+.
    df['step_dist'] = np.where(
        df['raw_step_dist'] <= MAX_STEP_DIST_M,
        df['raw_step_dist'],
        0.0
    )
    glitch_count = (df['raw_step_dist'] > MAX_STEP_DIST_M).sum()
    if glitch_count > 0:
        glitch_dist = (df['raw_step_dist'] - df['step_dist']).sum()
        print(
            f"[Distance] Discarded {glitch_count} glitch jumps "
            f"({glitch_dist:.1f}m of tracking noise removed)"
        )

    dist = df.groupby(['tracker_id', 'team_id', 'role'])['step_dist'].sum().reset_index()
    dist.columns = ['tracker_id', 'team_id', 'role', 'distance_m']
    dist['distance_m'] = dist['distance_m'].round(1)
    return dist.sort_values('distance_m', ascending=False)


def calc_speed_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df[df['speed_kmh'] > 0].groupby(
        ['tracker_id', 'team_id', 'role']
    )['speed_kmh'].agg(['max', 'mean']).reset_index()
    stats.columns = ['tracker_id', 'team_id', 'role', 'max_speed', 'avg_speed']
    stats['max_speed'] = stats['max_speed'].round(1)
    stats['avg_speed'] = stats['avg_speed'].round(1)
    return stats.sort_values('max_speed', ascending=False)


def calc_team_stats(df: pd.DataFrame, dist_df: pd.DataFrame) -> pd.DataFrame:
    speed_stats = df[df['speed_kmh'] > 0].groupby('team_id')['speed_kmh'].agg(
        avg_speed='mean', max_speed='max'
    ).round(2)
    dist_stats = dist_df.groupby('team_id')['distance_m'].agg(
        total_distance='sum', avg_distance='mean'
    ).round(1)
    return pd.concat([speed_stats, dist_stats], axis=1)


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------
def generate_text_report(df: pd.DataFrame, output_dir: str):
    possession = calc_possession(df)
    dist_df = calc_distance(df)
    speed_df = calc_speed_stats(df)
    team_stats = calc_team_stats(df, dist_df)

    total_frames = df['frame'].max()
    duration_sec = total_frames / FPS

    lines = []
    lines.append("=" * 50)
    lines.append("        SOCCER AI — MATCH REPORT")
    lines.append("=" * 50)
    lines.append(f"Clip duration : {duration_sec:.1f} seconds ({total_frames} frames)")
    lines.append(f"Players tracked: {df[df['role']=='player']['tracker_id'].nunique()}")
    lines.append("")

    lines.append("--- BALL POSSESSION ---")
    for tid, pct in possession.items():
        lines.append(f"  {TEAM_NAMES.get(tid, f'Team {tid}')}: {pct}%")
    lines.append("")

    lines.append("--- TEAM SUMMARY ---")
    for tid, row in team_stats.iterrows():
        if tid == 3:
            continue
        lines.append(f"  {TEAM_NAMES.get(tid, f'Team {tid}')}:")
        lines.append(f"    Avg speed      : {row['avg_speed']} km/h")
        lines.append(f"    Max speed      : {row['max_speed']} km/h")
        lines.append(f"    Total distance : {row['total_distance']} m")
        lines.append(f"    Avg per player : {row['avg_distance']} m")
    lines.append("")

    lines.append("--- TOP 5 DISTANCE COVERED ---")
    for _, row in dist_df[dist_df['role'] == 'player'].head(5).iterrows():
        lines.append(
            f"  #{int(row['tracker_id'])} ({TEAM_NAMES.get(int(row['team_id']), '?')})"
            f" — {row['distance_m']} m"
        )
    lines.append("")

    lines.append("--- TOP 5 MAX SPEED ---")
    for _, row in speed_df[speed_df['role'] == 'player'].head(5).iterrows():
        lines.append(
            f"  #{int(row['tracker_id'])} ({TEAM_NAMES.get(int(row['team_id']), '?')})"
            f" — {row['max_speed']} km/h (avg {row['avg_speed']} km/h)"
        )
    lines.append("")
    lines.append("=" * 50)

    report = "\n".join(lines)
    print(report)

    report_path = os.path.join(output_dir, "match_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n[Report] Saved → {report_path}")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def generate_charts(df: pd.DataFrame, output_dir: str):
    dist_df = calc_distance(df)
    possession = calc_possession(df)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Soccer AI — Coaching Insights', color='white', fontsize=16, fontweight='bold')

    # 1. Possession pie chart
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a2e')
    poss_teams = {k: v for k, v in possession.items() if k != 3}
    colors = [TEAM_COLORS.get(t, '#888888') for t in poss_teams.keys()]
    labels = [f"{TEAM_NAMES.get(t, f'Team {t}')}\n{v}%" for t, v in poss_teams.items()]
    ax.pie(poss_teams.values(), labels=labels, colors=colors,
           textprops={'color': 'white', 'fontsize': 11},
           wedgeprops={'edgecolor': '#1a1a2e', 'linewidth': 2})
    ax.set_title('Ball Possession', color='white', fontsize=12)

    # 2. Distance covered per player (top 10)
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a2e')
    top_dist = dist_df[dist_df['role'] == 'player'].head(10)
    bar_colors = [TEAM_COLORS.get(int(t), '#888888') for t in top_dist['team_id']]
    bars = ax.barh(
        [f"#{int(r['tracker_id'])} {TEAM_NAMES.get(int(r['team_id']), '')}"
         for _, r in top_dist.iterrows()],
        top_dist['distance_m'],
        color=bar_colors
    )
    ax.set_xlabel('Distance (m)', color='white')
    ax.set_title('Distance Covered — Top 10 Players', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.set_facecolor('#1a1a2e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.invert_yaxis()

    # 3. Speed distribution per team
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a2e')
    for tid in [0, 1]:
        speeds = df[(df['team_id'] == tid) & (df['speed_kmh'] > 0)]['speed_kmh']
        ax.hist(speeds, bins=30, alpha=0.6,
                color=TEAM_COLORS.get(tid, '#888888'),
                label=TEAM_NAMES.get(tid, f'Team {tid}'))
    ax.set_xlabel('Speed (km/h)', color='white')
    ax.set_ylabel('Frequency', color='white')
    ax.set_title('Speed Distribution by Team', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    # 4. Average speed per player (top 10)
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a2e')
    speed_df = df[df['speed_kmh'] > 0].groupby(
        ['tracker_id', 'team_id']
    )['speed_kmh'].mean().reset_index()
    speed_df = speed_df[speed_df['team_id'].isin([0, 1])].sort_values(
        'speed_kmh', ascending=False).head(10)
    bar_colors2 = [TEAM_COLORS.get(int(t), '#888888') for t in speed_df['team_id']]
    ax.barh(
        [f"#{int(r['tracker_id'])} {TEAM_NAMES.get(int(r['team_id']), '')}"
         for _, r in speed_df.iterrows()],
        speed_df['speed_kmh'],
        color=bar_colors2
    )
    ax.set_xlabel('Avg Speed (km/h)', color='white')
    ax.set_title('Average Speed — Top 10 Players', color='white', fontsize=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.invert_yaxis()

    # Legend
    patches = [mpatches.Patch(color=TEAM_COLORS[t], label=TEAM_NAMES[t]) for t in [0, 1]]
    fig.legend(handles=patches, loc='lower center', ncol=2,
               facecolor='#1a1a2e', labelcolor='white', fontsize=11)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    chart_path = os.path.join(output_dir, "insights_charts.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Charts] Saved → {chart_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Soccer AI Coaching Insights')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to tracking-fixed.csv')
    parser.add_argument('--output_dir', type=str, default='data/insights',
                        help='Directory to save report and charts')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_and_clean(args.csv_path)

    generate_text_report(df, args.output_dir)
    generate_charts(df, args.output_dir)
    print("\n[Done] All insights saved to", args.output_dir)


if __name__ == '__main__':
    main()