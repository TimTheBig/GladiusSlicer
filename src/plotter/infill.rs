use crate::plotter::monotone::get_monotone_sections;
use crate::settings::LayerSettings;
use crate::types::{Move, MoveChain, MoveType};

use serde::{Deserialize, Serialize};

use geo::prelude::*;
use geo::*;
use geo_clipper::*;

pub trait SolidInfillFill {
    fn fill(&self, filepath: &str) -> Vec<MoveChain>;
}

pub trait PartialInfillFill {
    fn fill(&self, filepath: &str) -> Vec<MoveChain>;
}

pub enum SolidInfillsTypes {
    Rectilinear,
}

pub fn linear_fill_polygon(
    poly: &Polygon<f64>,
    settings: &LayerSettings,
    fill_type: MoveType,
    angle: f64,
) -> Vec<MoveChain> {
    let rotate_poly = poly.rotate_around_point(angle, Point(Coordinate::zero()));

    let mut new_moves =
        spaced_fill_polygon(&rotate_poly, settings, fill_type, settings.layer_width, 0.0);

    for chain in new_moves.iter_mut() {
        chain.rotate(-angle.to_radians());
    }

    new_moves
}

pub fn partial_linear_fill_polygon(
    poly: &Polygon<f64>,
    settings: &LayerSettings,
    fill_type: MoveType,
    spacing: f64,
    angle: f64,
    offset: f64,
) -> Vec<MoveChain> {
    let rotate_poly = poly.rotate_around_point(angle, Point(Coordinate::zero()));

    let mut new_moves = spaced_fill_polygon(&rotate_poly, settings, fill_type, spacing, offset);

    for chain in new_moves.iter_mut() {
        chain.rotate(-angle.to_radians());
    }

    new_moves
}

pub fn solid_infill_polygon(
    poly: &Polygon<f64>,
    settings: &LayerSettings,
    fill_type: MoveType,
    layer_count: usize,
    layer_height: f64,
) -> Vec<MoveChain> {
    let angle = 45.0 + (120_f64) * layer_count as f64;

    linear_fill_polygon(poly, settings, fill_type, angle)
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum PartialInfillTypes {
    Linear,
    Rectilinear,
    Triangle,
    Cubic,
}

pub fn partial_infill_polygon(
    poly: &Polygon<f64>,
    settings: &LayerSettings,
    fill_ratio: f64,
    layer_count: usize,
    layer_height: f64,
) -> Vec<MoveChain> {
    match settings.infill_type {
        PartialInfillTypes::Linear => partial_linear_fill_polygon(
            poly,
            settings,
            MoveType::Infill,
            settings.layer_width / fill_ratio,
            0.0,
            0.0,
        ),
        PartialInfillTypes::Rectilinear => {
            let mut fill = partial_linear_fill_polygon(
                poly,
                settings,
                MoveType::Infill,
                2.0 * settings.layer_width / fill_ratio,
                45.0,
                0.0,
            );
            fill.append(&mut partial_linear_fill_polygon(
                poly,
                settings,
                MoveType::Infill,
                2.0 * settings.layer_width / fill_ratio,
                135.0,
                0.0,
            ));
            fill
        }
        PartialInfillTypes::Triangle => {
            let mut fill = partial_linear_fill_polygon(
                poly,
                settings,
                MoveType::Infill,
                3.0 * settings.layer_width / fill_ratio,
                45.0,
                0.0,
            );
            fill.append(&mut partial_linear_fill_polygon(
                poly,
                settings,
                MoveType::Infill,
                3.0 * settings.layer_width / fill_ratio,
                45.0 + 60.0,
                0.0,
            ));
            fill.append(&mut partial_linear_fill_polygon(
                poly,
                settings,
                MoveType::Infill,
                3.0 * settings.layer_width / fill_ratio,
                45.0 + 120.0,
                0.0,
            ));
            fill
        }
        PartialInfillTypes::Cubic => {
            let mut fill = partial_linear_fill_polygon(
                poly,
                settings,
                MoveType::Infill,
                3.0 * settings.layer_width / fill_ratio,
                45.0,
                layer_height / std::f64::consts::SQRT_2,
            );
            fill.append(&mut partial_linear_fill_polygon(
                poly,
                settings,
                MoveType::Infill,
                3.0 * settings.layer_width / fill_ratio,
                45.0 + 120.0,
                layer_height / std::f64::consts::SQRT_2,
            ));
            fill.append(&mut partial_linear_fill_polygon(
                poly,
                settings,
                MoveType::Infill,
                3.0 * settings.layer_width / fill_ratio,
                45.0 + 240.0,
                layer_height / std::f64::consts::SQRT_2,
            ));
            fill
        }
    }
}

pub fn spaced_fill_polygon(
    poly: &Polygon<f64>,
    settings: &LayerSettings,
    fill_type: MoveType,
    spacing: f64,
    offset: f64,
) -> Vec<MoveChain> {
    poly.offset(
        ((-settings.layer_width / 2.0) * (1.0 - settings.infill_perimeter_overlap_percentage))
            + (settings.layer_width / 2.0),
        JoinType::Square,
        EndType::ClosedPolygon,
        100000.0,
    )
    .iter()
    .filter(|poly| poly.unsigned_area() > 1.0)
    .map(|poly| {
        get_monotone_sections(poly)
            .iter()
            .filter_map(|section| {
                let mut current_y = (((section.left_chain[0].y + offset) / spacing).floor()
                    - (offset / spacing))
                    * spacing;

                let mut orient = true;

                let mut start_point = None;

                let mut line_change = true;

                let mut left_index = 0;
                let mut right_index = 0;

                let mut moves = vec![];

                loop {
                    while left_index < section.left_chain.len()
                        && section.left_chain[left_index].y > current_y
                    {
                        left_index += 1;
                        line_change = true;
                    }

                    if left_index == section.left_chain.len() {
                        break;
                    }

                    while right_index < section.right_chain.len()
                        && section.right_chain[right_index].y > current_y
                    {
                        right_index += 1;
                        line_change = true;
                    }

                    if right_index == section.right_chain.len() {
                        break;
                    }

                    let left_top = section.left_chain[left_index - 1];
                    let left_bot = section.left_chain[left_index];
                    let right_top = section.right_chain[right_index - 1];
                    let right_bot = section.right_chain[right_index];

                    let left_point = point_lerp(&left_top, &left_bot, current_y);
                    let right_point = point_lerp(&right_top, &right_bot, current_y);

                    start_point = start_point.or(Some(Coordinate {
                        x: left_point.x,
                        y: current_y,
                    }));

                    if orient {
                        moves.push(Move {
                            end: Coordinate {
                                x: left_point.x,
                                y: current_y,
                            },
                            move_type: if line_change {
                                MoveType::Travel
                            } else {
                                fill_type
                            },
                            width: settings.layer_width,
                        });

                        moves.push(Move {
                            end: Coordinate {
                                x: right_point.x,
                                y: current_y,
                            },
                            move_type: fill_type,
                            width: settings.layer_width,
                        });
                    } else {
                        moves.push(Move {
                            end: Coordinate {
                                x: right_point.x,
                                y: current_y,
                            },
                            move_type: if line_change {
                                MoveType::Travel
                            } else {
                                fill_type
                            },
                            width: settings.layer_width,
                        });

                        moves.push(Move {
                            end: Coordinate {
                                x: left_point.x,
                                y: current_y,
                            },
                            move_type: fill_type,
                            width: settings.layer_width,
                        });
                    }

                    orient = !orient;
                    current_y -= spacing;
                    line_change = false;
                }

                start_point.map(|start_point| MoveChain { start_point, moves })
            })
            .collect::<Vec<_>>()
            .into_iter()
    })
    .flatten()
    .collect()
}

#[inline]
fn point_lerp(a: &Coordinate<f64>, b: &Coordinate<f64>, y: f64) -> Coordinate<f64> {
    Coordinate {
        x: lerp(a.x, b.x, (y - a.y) / (b.y - a.y)),
        y,
    }
}

#[inline]
fn lerp(a: f64, b: f64, f: f64) -> f64 {
    a + f * (b - a)
}