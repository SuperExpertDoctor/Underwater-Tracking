from underwater_tracking.planning.plan_stability import rectangle_iou


def test_rectangle_iou_measures_region_change_in_the_shared_global_grid() -> None:
    assert rectangle_iou((0.0, 2000.0, 0.0, 2000.0), (1000.0, 3000.0, 0.0, 2000.0)) == 1 / 3
