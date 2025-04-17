use geo_3d::{Coord, Distance, Length, Line, Vector3DOps};
use gladius_shared::settings::Settings;
use gladius_shared::types::{Command, RetractionType, StateChange};
use itertools::Itertools;

/// Remove [`Command`]s that don't acheve anything
pub fn unary_optimizer(cmds: &mut Vec<Command>) {
    cmds.retain(|cmd| match cmd {
        Command::MoveTo { .. } => true,
        Command::MoveAndExtrude { start, end, .. } => start != end,
        Command::LayerChange { .. } | Command::ChangeObject { .. } => true,
        Command::SetState { new_state } => {
            !(new_state.acceleration.is_none()
                && new_state.movement_speed.is_none()
                && new_state.fan_speed.is_none()
                && new_state.retract == RetractionType::NoRetract
                && new_state.extruder_temp.is_none()
                && new_state.bed_temp.is_none())
        }
        Command::Delay { msec } => *msec != 0,
        Command::Arc {
            start, end, center, ..
        } => start != end || start != center,
        Command::NoAction => false,
    });
}

/// Merge consecutive commands, that have the same end result when merged
pub fn binary_optimizer(cmds: &mut Vec<Command>, settings: &Settings) {
    let mut current_pos = Coord::zero();

    *cmds = cmds
        .drain(..)
        .coalesce(move |first, second| {
            match (first.clone(), second.clone()) {
                (
                    Command::MoveAndExtrude {
                        start: f_start,
                        end: f_end,
                        thickness: f_thick,
                        width: f_width,
                    },
                    Command::MoveAndExtrude {
                        start: s_start,
                        end: s_end,
                        thickness: s_thick,
                        width: s_width,
                    },
                ) => {
                    current_pos = s_end;

                    if f_end == s_start && s_width == f_width && s_thick == f_thick {
                        // todo check if this checks co-planar or co-linear(correct)
                        let determinant = ((f_start.x - s_start.x) * ((s_start.y - s_end.y) * (f_start.z - s_start.z) - (f_start.y - s_start.y) * (s_start.z - s_end.z))
                        - (f_start.y - s_start.y) * ((s_start.x - s_end.x) * (f_start.z - s_start.z) - (f_start.x - s_start.x) * (s_start.z - s_end.z))
                        + (f_start.z - s_start.z) * ((s_start.x - s_end.x) * (f_start.y - s_start.y) - (f_start.x - s_start.x) * (s_start.y - s_end.y)))
                        .abs();

                        if determinant < 0.00001 {
                            // Collinear, merge commands
                            return Ok(Command::MoveAndExtrude {
                                start: f_start,
                                end: s_end,
                                thickness: f_thick,
                                width: s_width,
                            });
                        }
                    }
                }
                (Command::MoveTo { .. }, Command::MoveTo { end: s_end }) => {
                    current_pos = s_end;
                    return Ok(Command::MoveTo { end: s_end });
                }
                (Command::Delay { msec: t1 }, Command::Delay { msec: t2 }) => {
                    // merge back to back delays
                    return Ok(Command::Delay { msec: t1 + t2 });
                }
                (Command::ChangeObject { .. }, Command::ChangeObject { object }) => {
                    // skip an object change followed by another change
                    return Ok(Command::ChangeObject { object });
                }

                (
                    Command::SetState { new_state: f_state },
                    Command::SetState { new_state: s_state },
                ) => {
                    return Ok(Command::SetState {
                        new_state: f_state.combine(&s_state),
                    });
                }
                (
                    Command::SetState {
                        new_state: mut f_state,
                    },
                    Command::MoveTo { end },
                ) => {
                    if f_state.retract == RetractionType::Retract
                        && Line::new(current_pos, end).length()
                            < settings.minimum_retract_distance
                    {
                        current_pos = end;

                        // remove retract command
                        f_state.retract = RetractionType::NoRetract;

                        return Err((
                            Command::SetState { new_state: f_state },
                            Command::MoveTo { end },
                        ));
                    } else if let RetractionType::MoveRetract(_) = f_state.retract {
                        if Line::new(current_pos, end).length()
                            < settings.minimum_retract_distance
                        {
                            current_pos = end;

                            // remove retract command
                            f_state.retract = RetractionType::NoRetract;

                            return Err((
                                Command::SetState { new_state: f_state },
                                Command::MoveTo { end },
                            ));
                        }
                    } else {
                        current_pos = end;
                    }
                }
                (
                    _,
                    Command::MoveAndExtrude {
                        start: _s_start,
                        end: s_end,
                        ..
                    },
                ) => {
                    current_pos = s_end;
                }
                (_, Command::MoveTo { end: s_end }) => {
                    current_pos = s_end;
                }
                (_, _) => {}
            }

            Err((first, second))
        })
        .collect();
}

/// Simplify `SetState` commands to only be a diff from the last state
pub fn state_optomizer(cmds: &mut Vec<Command>) {
    let mut current_state = StateChange::default();

    for cmd_ptr in cmds {
        if let Command::SetState { new_state } = cmd_ptr {
            *new_state = current_state.state_diff(new_state);
        }
    }
}

#[allow(unused)]
pub fn arc_optomizer(cmds: &mut [Command]) {
    let mut ranges = vec![];

    for (wt, group) in &cmds.iter().enumerate().chunk_by(|cmd| {
        if let Command::MoveAndExtrude { thickness, width, .. } = cmd.1 {
            Some((thickness, width))
        } else {
            None
        }
    }) {
        if let Some((thickness, width)) = wt {
            let mut current_center = (0.0, 0.0, 0.0);
            let mut current_radius = 0.0;
            let mut current_chain = 0;

            let mut last_pos = 0;
            let mut group_peek = group.peekable();
            let mut start_pos = group_peek.peek().expect("validated aboive").0;

            for (pos, center, radius) in group_peek
                // commands -> lines
                .map(|(pos, cmd)| {
                    if let Command::MoveAndExtrude { start, end, .. } = cmd {
                        (pos, (start, end))
                    } else {
                        unreachable!()
                    }
                })
                // lines -> bisector
                .tuple_windows::<(
                    (usize, (&Coord<f64>, &Coord<f64>)),
                    (usize, (&Coord<f64>, &Coord<f64>)),
                )>()
                .map(|((pos, l1), (_, l2))| {
                    // todo try to avoid copy
                    (pos, line_bisector(*l1.0, *l1.1, *l2.1))
                })
                // bisector -> center, radius
                .tuple_windows::<(
                    (usize, (Coord<f64>, Coord<f64>)),
                    (usize, (Coord<f64>, Coord<f64>)),
                )>()
                .filter_map(|((pos, (p1, n1)), (_, (p2, n2)))| {
                    ray_ray_intersection(Line::new(p1, p2), Line::new(n1, n2))
                        .map(|center| (pos, center.x_y_z(), center.distance(p1)))
                })
            {
                last_pos = pos;

                if (radius - current_radius).abs() < 1.1
                    && (center.0 - current_center.0).abs() < 1.1
                    && (center.1 - current_center.1).abs() < 1.1
                {
                    current_chain += 1;
                    continue;
                }

                if current_chain > 5 {
                    ranges.push((center, (start_pos..=pos), *thickness, *width));
                }

                current_center = center;
                current_radius = radius;
                current_chain = 1;
                start_pos = pos;
            }

            if current_chain > 5 {
                ranges.push((
                    current_center,
                    (start_pos..=last_pos + 2),
                    *thickness,
                    *width,
                ));
            }
        }
    }

    for (center, mut range, thickness, width) in ranges {
        // todo fix
        let start = match cmds[*range.start()] {
            Command::MoveAndExtrude { start, .. } => start,
            _ => continue,
        };
        let end = match cmds[*range.end()] {
            Command::MoveAndExtrude { end, .. } => end,
            _ => continue,
        };

        for i in range.by_ref() {
            cmds[i] = Command::NoAction;
        }

        cmds[*range.start()] = Command::Arc {
            start,
            end,
            clockwise: true,
            center: Coord {
                x: center.0,
                y: center.1,
                z: center.2,
            },
            thickness,
            width,
        };
    }
}

#[allow(unused)]
fn line_bisector(p0: Coord<f64>, p1: Coord<f64>, p2: Coord<f64>) -> (Coord<f64>, Coord<f64>) {
    let l1_len = p0.distance(p1);
    let l2_len = p1.distance(p2);

    let l1_unit = (p1 - p0) / -l1_len;
    let l2_unit = (p1 - p2) / -l2_len;

    let dir = l1_unit + l2_unit;

    (/* ray_start */ p1, dir)
}

/// Computes the intersection point between a finite line segment and an infinite ray in 3D space.
///
/// The function solves for the intersection using parametric equations and vector cross products.\
/// It ensures the intersection lies within the segment and in the forward direction of the ray.
///
/// ## Arguments
/// - `s`: A finite `Line<f64>`.
/// - `d`: A `Line<f64>` that extends infinitely.
///
/// ## Returns
/// - `Some(Coord<f64>)` if an intersection is found.
/// - `None` if the line segment and ray do not intersect.
///
/// ## Notes
/// - If the segment and ray are **parallel**, the function returns `None`.
/// - If the intersection occurs **behind the ray's origin**, it is ignored.
///
/// # Example
/// ```
/// # use geo_3d::{Line, Ray, Coord};
/// let segment = Line {
///     start: Coord { x: 1.0, y: 1.0, z: 1.0 },
///     end: Coord { x: 4.0, y: 4.0, z: 4.0 },
/// };
/// let ray = Line {
///     start: Coord { x: 2.0, y: 2.0, z: 0.0 }, // origin
///     end: Coord { x: 0.0, y: 0.0, z: 1.0 }, // Ray extends infinitely along z-axis
/// };
///
/// let intersection = ray_ray_intersection(segment, ray);
/// assert_eq!(intersection, Some(Coord { x: 2.0, y: 2.0, z: 2.0 }));
/// ```
#[allow(unused)]
fn ray_ray_intersection(s: Line<f64>, d: Line<f64>) -> Option<Coord<f64>> {
    const EPSILON: f64 = 1e-9;

    // Direction vector of segment
    let s_dir = s.end - s.start;
    // Direction vector of the ray
    let magnitude = (d.end.x.powi(2) + d.end.y.powi(2) + d.end.z.powi(2)).sqrt();
    let unit_direction = d.end / magnitude;
    let d_dir = unit_direction;

    // Vector between segment start and ray origin
    let w0 = s.start - d.start;

    // Compute cross products
    let cross_sd = s_dir.cross(d_dir);
    let cross_sd_magnitude_squared = cross_sd.magnitude_squared();
    let cross_wd = w0.cross(d_dir);

    // Assert that direction vectors are not zero
    debug_assert!(s_dir.magnitude() > EPSILON, "Segment has zero length!");
    debug_assert!(d_dir.magnitude() > EPSILON, "Ray direction cannot be zero!");

    // Check if lines are parallel (cross product is zero vector)
    if cross_sd_magnitude_squared.sqrt() < EPSILON {
        return None;
    }

    // Solve for parameters t (on segment) and u (on ray)
    let t = cross_wd.dot(cross_sd) / cross_sd_magnitude_squared;
    let u = w0.cross(s_dir).dot(cross_sd) / cross_sd_magnitude_squared;

    // Ensure intersection is within segment bounds (0 ≤ t ≤ 1) and ray extends forward (u ≥ 0)
    if (-EPSILON..=1.0 + EPSILON).contains(&t) && u >= -EPSILON {
        let intersection_point = s.start + s_dir * t;
        return Some(intersection_point);
    }

    None // No valid intersection
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersection() {
        let segment = Line {
            start: Coord { x: 1.0, y: 1.0, z: 1.0 },
            end: Coord { x: 4.0, y: 4.0, z: 4.0 },
        };
        let ray = Line {
            start: Coord { x: 0.0, y: 0.0, z: 0.0 },
            end: Coord { x: 1.0, y: 1.0, z: 1.0 },
        };
        let expected = Some(Coord { x: 1.0, y: 1.0, z: 1.0 });

        assert_eq!(ray_ray_intersection(segment, ray), expected);
    }

    #[test]
    fn test_no_intersection() {
        let segment = Line {
            start: Coord { x: 2.0, y: 2.0, z: 2.0 },
            end: Coord { x: 4.0, y: 4.0, z: 4.0 },
        };
        let ray = Line {
            start: Coord { x: 0.0, y: 0.0, z: 0.0 },
            end: Coord { x: -1.0, y: -1.0, z: -1.0 },
        };

        assert_eq!(ray_ray_intersection(segment, ray), None);
    }

    #[test]
    fn test_parallel_no_intersection() {
        let segment = Line {
            start: Coord { x: 1.0, y: 1.0, z: 1.0 },
            end: Coord { x: 3.0, y: 3.0, z: 3.0 },
        };
        let ray = Line {
            start: Coord { x: 0.0, y: 0.0, z: 0.0 },
            end: Coord { x: 2.0, y: 2.0, z: 2.0 },
        };

        assert_eq!(ray_ray_intersection(segment, ray), None);
    }

    #[test]
    fn basic_line_bisector() {
        let (center, dir) = line_bisector(
            Coord { x: 0.0, y: 0.0, z: 0.0 },
            Coord { x: 1.0, y: 1.0, z: 1.0 },
            Coord { x: 2.0, y: 0.0, z: 2.0 },
        );

        assert_eq!(center, Coord { x: 1.0, y: 1.0, z: 1.0 });
        assert_eq!(dir.x, 0.0);
        assert!(dir.y < 0.0);

        let (center, dir) = line_bisector(
            Coord { x: 2.0, y: 0.0, z: 2.0 },
            Coord { x: 1.0, y: 1.0, z: 1.0 },
            Coord { x: 0.0, y: 0.0, z: 0.0 },
        );

        assert_eq!(center, Coord { x: 1.0, y: 1.0, z: 1.0 });
        assert_eq!(dir.x, 0.0);
        assert!(dir.y < 0.0);

        let (center, dir) = line_bisector(
            Coord { x: 0.0, y: 0.0, z: 0.0 },
            Coord { x: 1.0, y: 1.0, z: 1.0 },
            Coord { x: -2.0, y: 4.0, z: -2.0 },
        );

        assert_eq!(center, Coord { x: 1.0, y: 1.0, z: 1.0 });
        assert!(dir.y - 0.0 < 0.000001);
        assert!(dir.x < 0.0);

        let (center, dir) = line_bisector(
            Coord { x: 0.0, y: 0.0, z: 0.0 },
            Coord { x: 1.0, y: 0.0, z: 1.0 },
            Coord { x: 1.0, y: 1.0, z: 1.0 },
        );

        assert_eq!(center, Coord { x: 1.0, y: 0.0, z: 1.0 });
        assert_eq!(dir.y, -dir.x);
    }

    #[test]
    fn basic_ray_ray() {
        let center = ray_ray_intersection(
            Line {
                start: Coord { x: 0.0, y: 0.0, z: 0.0 },
                end: Coord { x: 2.0, y: 0.0, z: 2.0 },
            },
            Line {
                start: Coord { x: -1.0, y: 1.0, z: -1.0 },
                end: Coord { x: 1.0, y: 1.0, z: 1.0 },
            }
        );
        assert_eq!(center, Some(Coord { x: 2.0, y: 0.0, z: 2.0 }));

        let center = ray_ray_intersection(
            Line {
                start: Coord { x: 0.0, y: 3.0, z: 0.0 },
                end: Coord { x: 2.0, y: 0.0, z: 0.0 },
            },
            Line {
                start: Coord { x: 3.0, y: 4.0, z: 0.0 },
                end: Coord { x: 5.0, y: 1.0, z: 0.0 },
            },
        );
        assert_eq!(center, Some(Coord { x: 5.0, y: 4.0, z: 0.0 }));

        let center = ray_ray_intersection(
            Line {
                start: Coord { x: 1.0, y: 3.0, z: 0.0 },
                end: Coord { x: 0.0, y: -2.0, z: 0.0 },
            },
            Line {
                start: Coord { x: 2.0, y: 3.0, z: 0.0 },
                end: Coord { x: 0.10, y: -0.20, z: 0.0 },
            },
        );
        assert_eq!(center, Some(Coord { x: 2.0, y: 1.0, z: 0.0 }));
    }

    #[test]
    fn arc_optomizer_test() {
        let mut commands = (0..200)
            .map(|a| {
                let r = a as f64 / 100.0;
                let x = r.cos();
                let y = r.sin();
                let z = r.cos();
                Coord { x, y, z }
            })
            .tuple_windows::<(Coord<f64>, Coord<f64>)>()
            .map(|(start, end)| Command::MoveAndExtrude {
                start,
                end,
                thickness: 0.3,
                width: 0.4,
            })
            .collect::<Vec<Command>>();

        arc_optomizer(&mut commands);
        unary_optimizer(&mut commands);

        assert_eq!(commands.len(), 1);
        if let Command::Arc {
            start,
            center,
            width,
            thickness,
            ..
        } = commands[0] {
            assert_eq!(start, Coord { x: 1.0, y: 0.0, z: 1.0 });
            assert_eq!(center, Coord { x: 0.0, y: 0.0, z: 0.0 });
            assert_eq!(width, 0.4);
            assert_eq!(thickness, 0.3);
        } else {
            panic!("Command should be an arc")
        }
    }

    #[test]
    fn arc_optomizer_test_adv() {
        let mut commands = vec![Command::Delay { msec: 1000 }];

        commands.extend(
            (0..200)
                .map(|a| {
                    let r = a as f64 / 100.0;
                    let x = r.cos();
                    let y = r.sin();
                    let z = r.tan();
                    Coord { x, y, z }
                })
                .tuple_windows::<(Coord<f64>, Coord<f64>)>()
                .map(|(start, end)| Command::MoveAndExtrude {
                    start,
                    end,
                    thickness: 0.3,
                    width: 0.4,
                }),
        );

        commands.push(Command::Delay { msec: 1000 });

        arc_optomizer(&mut commands);
        unary_optimizer(&mut commands);

        assert_eq!(commands.len(), 3);
        if let Command::Arc {
            start,
            center,
            width,
            thickness,
            ..
        } = commands[1] {
            assert_eq!(start, Coord { x: 1.0, y: 0.0, z: 1.0 });
            assert_eq!(center, Coord { x: 0.0, y: 0.0, z: 0.0 });
            assert_eq!(width, 0.4);
            assert_eq!(thickness, 0.3);
        } else {
            panic!("Command should be an arc")
        }
    }
}
