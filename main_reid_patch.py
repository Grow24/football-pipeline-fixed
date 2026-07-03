# ─────────────────────────────────────────────────────────────────────────────
# PATCH: Add SigLIP Re-ID to main-2.py
# Apply these changes in order. Lines marked [ADD] are new; [CHANGE] replace existing.
# ─────────────────────────────────────────────────────────────────────────────


# ══ SECTION 1: Imports (top of file) ════════════════════════════════════════
# [ADD] after your existing imports:
from siglip_reid import ReIDTracker, ReIDConfig


# ══ SECTION 2: Init (after device is set, before cap.open) ══════════════════
# [ADD]:
reid_cfg = ReIDConfig(
    match_threshold=0.82,   # raise to 0.87 if you get false merges
    gallery_size=8,
    max_lost_frames=90,
    correct_team=True,
)
reid_tracker = ReIDTracker(device=str(device), cfg=reid_cfg)


# ══ SECTION 3: Frame loop — after ByteTrack gives you tracks ════════════════
# Your current code probably looks like:
#   tracks = byte_tracker.update(...)
#   for track in tracks:
#       tracker_id = track.track_id
#       ...

# [ADD] right after the byte_tracker.update() line:
#
#   # Convert supervision Detections to list-of-dicts for ReID
#   track_dicts = [
#       {
#           "tracker_id": int(tid),
#           "bbox_xyxy": list(box),
#           "team_id": int(team_ids[i]) if team_ids is not None else None,
#       }
#       for i, (tid, box) in enumerate(zip(tracker_ids, boxes_xyxy))
#   ]
#   track_dicts = reid_tracker.update(frame, track_dicts)
#
#   # Build a lookup so the rest of your loop can use canonical_id
#   canonical_lookup = {t["tracker_id"]: t["canonical_id"] for t in track_dicts}

# [CHANGE] wherever you write tracker_id to CSV / heatmap / display:
#   tracker_id  →  canonical_lookup.get(tracker_id, tracker_id)


# ══ SECTION 4: CSV write — tracking.csv row ══════════════════════════════════
# [CHANGE] your csv_writer.writerow to include canonical_id:
#   csv_writer.writerow({
#       ...existing fields...,
#       "canonical_id": canonical_lookup.get(tracker_id, tracker_id),
#   })
# Add "canonical_id" to your fieldnames list at the top too.


# ══ SECTION 5: End-of-video stats ════════════════════════════════════════════
# [ADD] before cap.release():
print("\n[ReID] Final stats:", reid_tracker.get_stats())
fragmentation_reduction = (
    reid_tracker.get_stats()["new_registrations"]
    / max(1, reid_tracker.get_stats()["total_canonical_ids"])
)


# ══ SECTION 6: Offline CSV fix (optional, no re-run needed) ═════════════════
# If you already have a tracking.csv and just want to patch IDs:
#
#   from siglip_reid import remap_csv
#   remap_csv(
#       tracking_csv="output/tracking.csv",
#       output_csv="output/tracking_reid.csv",
#   )
#
# This uses IoU-based spatial linking (no GPU needed) as a lightweight fix.


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG TUNING GUIDE
# ─────────────────────────────────────────────────────────────────────────────
#
# match_threshold:
#   0.75  → more aggressive merging (risk: wrong player merged)
#   0.82  → balanced (default)
#   0.87  → conservative (risk: same player gets two canonical IDs)
#
# max_lost_frames:
#   30    → only re-ID players lost < 1s (tight)
#   90    → up to 3s occlusion (default)
#   150   → up to 5s (useful for camera cuts)
#
# correct_team=True:
#   Prevents merging player from Team A with someone from Team B.
#   Requires team_id to be populated on track dicts.
#   Set False if your team classifier is unreliable.
#
# gallery_size=8:
#   Keeps last 8 crops per player. Higher = better recall, slower.
