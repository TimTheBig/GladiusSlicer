use geo_3d::{coord, Vector3DOps};
use crate::optimizer::{binary_optimizer, state_optomizer, unary_optimizer};
use crate::{Command, HashMap, Itertools, OrderedFloat, RetractionType, Settings};

pub trait CommandPass {
    fn pass(cmds: &mut Vec<Command>, settings: &Settings);
}

pub struct OptimizePass;

impl CommandPass for OptimizePass {
    fn pass(cmds: &mut Vec<Command>, settings: &Settings) {
        let mut size = cmds.len();

        loop {
            // run optimizations
            // arc_optomizer(cmds);
            state_optomizer(cmds);
            unary_optimizer(cmds);
            binary_optimizer(cmds, settings);

            // break if the size has not been changed
            if cmds.len() != size {
                size = cmds.len();
            } else {
                break;
            };
        }
    }
}

pub struct SlowDownLayerPass;

impl CommandPass for SlowDownLayerPass {
    // todo affect z
    fn pass(cmds: &mut Vec<Command>, settings: &Settings) {
        let mut layer_height = 0.0;
        // Slow down on small layers
        let mut current_speed = 0.0;
        let mut current_pos = &coord! { x: 0.0, y: 0.0, z: 0.0 };

        {
            let reduction: Vec<(f64, usize, usize)> = cmds
                .iter()
                .enumerate()
                .batching(|it| {
                    // map from speed to length at that speed
                    let mut map: HashMap<OrderedFloat<f64>, f64> = HashMap::new();
                    let mut non_move_time = 0.0;

                    let start_z_height = layer_height;
                    let mut return_none = false;

                    let mut start_index = None;
                    let mut end_index = 0;
                    while layer_height == start_z_height && !return_none {
                        if let Some((index, cmd)) = it.next() {
                            start_index = start_index.or(Some(index));
                            end_index = index;
                            match cmd {
                                Command::MoveTo { end } => {
                                    let d = (*end - *current_pos).magnitude();
                                    current_pos = end;
                                    if current_speed != 0.0 {
                                        non_move_time += d / current_speed;
                                    }
                                }
                                Command::MoveAndExtrude {
                                    start,
                                    end,
                                    width: _width,
                                    thickness: _thickness,
                                } => {
                                    let dis = (*end - *start).magnitude();
                                    current_pos = end;
                                    *map.entry(OrderedFloat(current_speed)).or_insert(0.0) += dis;
                                }
                                Command::SetState { new_state } => {
                                    if let Some(speed) = new_state.movement_speed {
                                        current_speed = speed;
                                    }

                                    if new_state.retract != RetractionType::NoRetract {
                                        non_move_time +=
                                            settings.retract_length / settings.retract_speed;
                                        non_move_time +=
                                            settings.retract_lift_z / settings.speed.travel;
                                    }
                                }
                                Command::Delay { msec } => {
                                    non_move_time += *msec as f64 / 1000.0;
                                }
                                Command::Arc {
                                    start, end, center, ..
                                } => {
                                    let cord_length = (*end - *start).magnitude();
                                    let radius = (*end - *center).magnitude();

                                    // Divide the chord length by double the radius.
                                    let t = cord_length / (2.0 * radius);

                                    // Find the inverse sine of the result (in radians).
                                    // Double the result of the inverse sine to get the central angle in radians.
                                    let central = t.asin() * 2.0;
                                    // Once you have the central angle in radians, multiply it by the radius to get the arc length.
                                    let extrusion_length = central * radius;

                                    current_pos = end;
                                    *map.entry(OrderedFloat(current_speed)).or_insert(0.0) +=
                                        extrusion_length;
                                }
                                Command::LayerChange { z, .. } => {
                                    layer_height = *z;
                                }
                                Command::NoAction | Command::ChangeObject { .. } => {}
                            }
                        } else {
                            return_none = true;
                        }
                    }

                    if return_none {
                        if map.is_empty() {
                            None
                        } else {
                            Some((
                                map,
                                non_move_time,
                                start_index
                                    .expect("For map to have values, start index must be set"),
                                end_index,
                            ))
                        }
                    } else {
                        Some((
                            map,
                            non_move_time,
                            start_index
                                .expect("For return_none to be false, start index must be set"),
                            end_index,
                        ))
                    }
                })
                .filter_map(|(map, time, start, end)| {
                    let mut total_time = time
                        + map
                            .iter()
                            .map(|(speed, len)| len / speed.into_inner())
                            .sum::<f64>();

                    let min_time = settings.fan.slow_down_threshold;
                    if total_time < min_time && !map.is_empty() {
                        let mut sorted = map.into_iter().collect::<Vec<(OrderedFloat<f64>, f64)>>();
                        sorted.sort_by(|a, b| a.0.cmp(&b.0));

                        let max_speed: f64;
                        loop {
                            let (speed, len) = sorted
                                .pop()
                                .expect("Because map isn't empty, sorted can't be empty");
                            let (top_speed, _) =
                                sorted.last().unwrap_or(&(OrderedFloat(0.000_001), 0.0));

                            if min_time - total_time
                                < (len / top_speed.into_inner()) - (len / speed.into_inner())
                            {
                                let second = min_time - total_time;
                                max_speed = (len * speed.into_inner())
                                    / (len + (second * speed.into_inner()));
                                break;
                            } else {
                                total_time +=
                                    (len / top_speed.into_inner()) - (len / speed.into_inner());
                            }
                        }
                        Some((max_speed, start, end))
                    } else {
                        None
                    }
                })
                .collect();

            reduction.into_iter().for_each(|(max_speed, start, end)| {
                for cmd in &mut cmds[start..end] {
                    if let Command::SetState { new_state } = cmd {
                        if let Some(speed) = &mut new_state.movement_speed {
                            if *speed != settings.speed.travel {
                                *speed = speed.min(max_speed).max(settings.fan.min_print_speed);
                            }
                        }
                    }
                }
            });
        }
    }
}
