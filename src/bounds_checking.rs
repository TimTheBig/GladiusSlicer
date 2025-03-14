use geo_3d::{contains::ContainsXY, MultiPolygon, Coord};
use gladius_shared::error::SlicerErrors;
use gladius_shared::settings::Settings;
use gladius_shared::types::{Command, IndexedTriangle, Vertex};
use itertools::Itertools;

/// Check if the point is in an excluded area
fn check_excluded(
    v_point: Coord,
    bed_exclude_areas: &Option<MultiPolygon>,
) -> Result<(), SlicerErrors> {
    for polygon in bed_exclude_areas.as_ref() {
        if polygon.contains_2d(&v_point) {
            return Err(SlicerErrors::InExcludeArea(polygon.clone()));
        }
    }

    Ok(())
}

/// Checks that all **vertices** in models are in the bed, acounting for the brim, and out of excluded areas.
pub fn check_model_bounds(
    models: &[(Vec<Vertex>, Vec<IndexedTriangle>)],
    settings: &Settings,
) -> Result<(), SlicerErrors> {
    let brim_width = settings.brim_width.unwrap_or(0.0);
    let shrink_distance = settings.layer_shrink_amount.unwrap_or(0.0);

    let total_offset = brim_width + shrink_distance;

    models
        .iter()
        .flat_map(|model| model.0.iter())
        .map(|v| {
            // Check if the point is in an excluded area
            check_excluded(v.0, &settings.bed_exclude_areas)?;

            if v.0.x < total_offset
                || v.0.y < total_offset
                || v.0.z < -0.00001
                || v.0.x > settings.print_x - total_offset
                || v.0.y > settings.print_y - total_offset
                || v.0.z > settings.print_z
            {
                Err(SlicerErrors::ModelOutsideBuildArea)
            } else {
                Ok(())
            }
        })
        .try_collect()
}

/// Checks all `Command`s to ashure they do not exceed the print diametions
pub fn check_moves_bounds(moves: &[Command], settings: &Settings) -> Result<(), SlicerErrors> {
    moves
        .iter()
        .map(|command| match command {
            Command::MoveTo { end, .. } | Command::MoveAndExtrude { end, .. } => {
                if end.x < 0.0
                    || end.x > settings.print_x
                    || end.y < 0.0
                    || end.y > settings.print_y
                    || end.z < 0.0
                    || end.z > settings.print_z
                {
                    Err(SlicerErrors::MovesOutsideBuildArea)
                } else {
                    Ok(())
                }
            }
            Command::LayerChange { z, .. } => {
                if *z > settings.print_z || *z < 0.0 {
                    Err(SlicerErrors::MovesOutsideBuildArea)
                } else {
                    Ok(())
                }
            }
            Command::Arc { .. } => {
                unimplemented!()
            }
            Command::SetState { .. }
            | Command::Delay { .. }
            | Command::NoAction
            | Command::ChangeObject { .. } => Ok(()),
        })
        .try_collect()
}

#[cfg(test)]
mod bounds_check_tests {
    use super::*;
    use geo_3d::{coord, LineString, Polygon};

    #[test]
    fn test_slice_with_model_in_excluded_area() {
        check_excluded(
            coord!(30.1, 58.6, 0.0),
            &Some(MultiPolygon::new(vec![Polygon::new(
                LineString::from(vec![(0.0, 0.0, 0.0), (256.0, 0.0, 0.0), (256.0, 256.0, 256.0), (0.0, 256.0, 0.0)]),
                Vec::new(),
            )])),
        )
        .unwrap_err();

        check_excluded(
            coord!(5.7, 8.4, 35.0),
            &Some(MultiPolygon::new(vec![Polygon::new(
                LineString::from(vec![(0.0, 0.0, 0.0), (2.0, 0.0, 2.0), (6.0, 5.0, 6.0), (0.0, 2.0, 0.0)]),
                Vec::new(),
            )])),
        )
        .unwrap();
    }
}
