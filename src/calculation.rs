use crate::{CalculatedValues, Command, Coord, RetractionType, Settings};
use geo_3d::Vector3DOps;

/// Compute the `CalculatedValues` of `moves`
pub fn calculate_values(moves: &[Command], settings: &Settings) -> CalculatedValues {
    let mut values = CalculatedValues {
        plastic_volume: 0.0,
        plastic_weight: 0.0,
        total_time: 0.0,
        plastic_length: 0.0,
    };

    let mut current_speed = 0.0;
    let mut current_pos = Coord { x: 0.0, y: 0.0, z: 0.0 };

    for cmd in moves {
        match cmd {
            Command::MoveTo { end } => {
                let dis = (*end - current_pos).magnitude();

                current_pos = *end;
                if current_speed != 0.0 {
                    values.total_time += dis / current_speed;
                }
            }
            Command::MoveAndExtrude {
                start,
                end,
                width,
                thickness,
            } => {
                let dis = (*end - *start).magnitude();

                current_pos = *end;
                values.total_time += dis / current_speed;

                values.plastic_volume += width * thickness * dis;
            }
            Command::SetState { new_state } => {
                if let Some(speed) = new_state.movement_speed {
                    current_speed = speed;
                }
                if new_state.retract != RetractionType::NoRetract {
                    values.total_time += settings.retract_length / settings.retract_speed;
                    values.total_time += settings.retract_lift_z / settings.speed.travel;
                }
            }
            Command::Delay { msec } => {
                values.total_time += *msec as f64 / 1000.0;
            }
            Command::Arc {
                start,
                end,
                center,
                width,
                thickness,
                ..
            } => {
                let coord_length = (*end - *start).magnitude();
                let radius = (*end - *center).magnitude();

                // Divide the chord length by double the radius.
                let t = coord_length / (2.0 * radius);

                // Find the inverse sine of the result (in radians).
                // Double the result of the inverse sine to get the central angle in radians.
                let central = t.asin() * 2.0;
                // Once you have the central angle in radians, multiply it by the radius to get the arc length.
                let extrusion_length = central * radius;

                values.total_time += extrusion_length / current_speed;

                values.plastic_volume += width * thickness * extrusion_length;
            }
            Command::NoAction | Command::LayerChange { .. } | Command::ChangeObject { .. } => {}
        }
    }

    values.plastic_weight = (values.plastic_volume / 1000.0) * settings.filament.density;
    values.plastic_length = values.plastic_volume / (
        std::f64::consts::PI * (settings.nozzle_diameter / 2.0) * (settings.nozzle_diameter / 2.0)
    );

    values
}
