#!/usr/bin/env python3
##
# @file regen_fig_overview.py
#
# @brief Regenerates fig_overview.png (paper Figure 1) with larger fonts,
#        using differential_drive_path_segment_combined.py's own cached
#        pipeline (_run_own_pipeline) so it does not re-run the slow w_e
#        sweep on every call. Must NOT reuse
#        differential_drive_comparison.py's pipeline: that module solves a
#        different corner scenario (WP_CORNER at (5,0), the paper's
#        evaluation scenario), while _fig1_segmented_path's reference
#        waypoint markers and inset zoom window are hardcoded for this
#        module's own scenario (WP_CORNER at (0,10)). Mixing the two
#        produced a Figure 1 with reference dots and an inset zoom pointing
#        at empty space while the actual trajectory was drawn elsewhere.
#
# @section author_doxygen_example Author(s)
# - Created by Tran Viet Thanh on 2026/08/20

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import differential_drive_comparison as dc
import differential_drive_path_segment_combined as psc


def main():
    dc._set_print_style()   # figure is drawn at its printed 9 cm width

    pipe = psc._run_own_pipeline()   # cached after first (slow) run

    psc._fig1_segmented_path(pipe['seg_info'], pipe['res_s1'], pipe['res_s2'],
                              pipe['res_corner_opt'])


if __name__ == '__main__':
    main()
