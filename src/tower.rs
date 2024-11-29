use crate::SlicerErrors;
use crate::utils::lerp;
use gladius_shared::types::{IndexedTriangle, Vertex};
use rayon::prelude::*;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

/*
    Rough algoritim

    build tower
        For each point store all edges and face connected to but above it

    progress up tower
*/


/// Calculate the **vertex**, the Line from `v_start` to `v_end` where
/// it intersects with the plane z
///
/// <div class="warning">If v_start.z == v_end.z then divide by 0</div>
///
/// ## Arguments
/// * `plane_height` - z height of the resulting point along the plane_normal
/// * `v_start` - Starting point of the line
/// * `v_end` - Ending point of the line
///
/// Calculate the intersection of a line with a plane defined by `normal` and `plane_height`
fn plane_intersection(plane_height: f64, v_start: &Vertex, v_end: &Vertex, normal: &Vertex) -> Vertex {
    // project on to plane
    let start_proj = v_start.dot(normal);
    let end_proj = v_end.dot(normal);

    let t = (plane_height - start_proj) / (end_proj - start_proj);
    debug_assert!(t <= 1.0);

    let x = lerp(v_start.x, v_end.x, t);
    let y = lerp(v_start.y, v_end.y, t);
    let z = lerp(v_start.z, v_end.z, t);

    Vertex { x, y, z }
}

/// A set of triangles and their associated vertices
pub struct TriangleTower {
    pub vertices: Vec<Vertex>,
    tower_vertices: Vec<TowerVertex>,
}

impl TriangleTower {
    /// Create a `TriangleTower` from **vertices** as leading or trailing edges and **triangles**
    /// The normal specifys the slicing plane's normal
    pub fn from_triangles_and_vertices(
        triangles: &[IndexedTriangle],
        vertices: Vec<Vertex>,
        normal: &Vertex,
    ) -> Result<Self, SlicerErrors> {
        let mut future_tower_vert: Vec<Vec<TriangleEvent>> =
            (0..vertices.len()).map(|_| Vec::new()).collect();

        // for each triangle add it to the tower
        for (triangle_index, index_tri) in triangles.iter().enumerate() {
            // index 0 is always lowest
            future_tower_vert[index_tri.verts[0]].push(TriangleEvent::MiddleVertex {
                trailing_edge: index_tri.verts[1],
                leading_edge: index_tri.verts[2],
                triangle: triangle_index,
            });

            // depending what the next vertex is its either leading or trailing
            if vertices[index_tri.verts[1]].dot(normal) < vertices[index_tri.verts[2]].dot(normal) {
                future_tower_vert[index_tri.verts[1]].push(TriangleEvent::TrailingEdge {
                    trailing_edge: index_tri.verts[2],
                    triangle: triangle_index,
                });
            } else {
                future_tower_vert[index_tri.verts[2]].push(TriangleEvent::LeadingEdge {
                    leading_edge: index_tri.verts[1],
                    triangle: triangle_index,
                });
            }
        }

        // for each triangle event, add it to the lowest vertex and
        // create a list of all vertices and there above edges
        let mut tower_vertices: Vec<TowerVertex> = future_tower_vert
            .into_iter()
            .enumerate()
            .map(|(index, events)| {
                let fragments = join_triangle_event(&events, index);
                TowerVertex {
                    start_index: index,
                    next_ring_fragments: fragments,
                    start_vert: vertices.get(index).expect("validated above").clone(),
                }
            })
            .collect();


        // ! this can case incomplet rings
        // Sort tower vertices lowest to highest based on their projection along the normal vector
        tower_vertices.sort_by(|a, b| {
            // project vertices on to plane and comp
            vertices[a.start_index].dot(normal)
                .partial_cmp(&vertices[b.start_index].dot(normal))
                .expect("STL ERROR: No Points should have NAN values")
        });

        Ok(Self {
            vertices,
            tower_vertices,
        })
    }


    pub fn get_height_of_vertex(&self, index: usize, plane_normal: &Vertex) -> f64 {
        if index >= self.tower_vertices.len() {
            f64::INFINITY
        } else {
            project_vertex_onto_plane(&self.vertices[self.tower_vertices[index].start_index], plane_normal)
        }
    }
}

/// A vecter of [`TowerRing`]s with a start index, made of triangles
#[derive(Debug)]
struct TowerVertex {
    pub next_ring_fragments: Vec<TowerRing>,
    pub start_index: usize,
    pub start_vert: Vertex,
}

impl PartialOrd for TowerVertex {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TowerVertex {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.start_vert
            .partial_cmp(&other.start_vert)
            .expect("NO_NAN")
            .reverse()
    }
}

impl Eq for TowerVertex {}

impl PartialEq for TowerVertex {
    fn eq(&self, other: &Self) -> bool {
        self.start_vert.eq(&other.start_vert)
    }
}

/// A list of **faces** and **edges**\
/// When complete it will have at least 3 [`TowerRingElement`]s and equal first and last elements.\
/// This needs to be checked with [`TowerRing::is_complete_ring`]
#[derive(Clone, Debug, PartialEq, Eq)]
struct TowerRing {
    elements: Vec<TowerRingElement>,
}

impl TowerRing {
    #[inline]
    /// Checks that the ring's vec is circuler
    fn is_complete_ring(&self) -> bool {
        self.elements.first() == self.elements.last() && self.elements.len() > 3
    }

    #[inline]
    /// Extend the elements of **first** with all but the first element of **second**
    fn join_rings_in_place(first: &mut TowerRing, second: TowerRing) {
        first.elements.extend(second.elements.into_iter().skip(1));
    }

    /// Split the `TowerRing` in to multiple at an edge
    fn split_on_edge(self, edge: usize) -> Vec<Self> {
        let mut new_ring = Vec::new();
        let mut frags = Vec::new();

        for e in self.elements {
            if let TowerRingElement::Edge { end_index, .. } = e {
                if end_index == edge {
                    frags.push(TowerRing { elements: new_ring });
                    new_ring = Vec::new();
                } else {
                    new_ring.push(e);
                }
            } else {
                new_ring.push(e);
            }
        }

        if frags.is_empty() {
            // add in the fragment
            frags.push(TowerRing { elements: new_ring });
        } else {
            // append to the beginning to prevent ophaned segments
            if frags[0].elements.is_empty() {
                frags[0].elements = new_ring;
            } else {
                new_ring.extend_from_slice(&frags[0].elements[1..]);
                frags[0].elements = new_ring;
            }
        }

        // ? should this be >= as "single" implise 1
        // remove all fragments that are single sized and faces. They ends with that vertex
        frags.retain(|frag| frag.elements.len() > 1);

        frags
    }
}

impl Display for TowerRing {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for e in &self.elements {
            write!(f, "{e} ")?;
        }

        Ok(())
    }
}

impl Ord for TowerRing {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.elements.first().cmp(&other.elements.first())
    }
}

impl PartialOrd for TowerRing {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Face or Edge of a [`TowerRing`]
#[derive(Clone, Debug, Eq)]
enum TowerRingElement {
    Face {
        triangle_index: usize,
    },
    Edge {
        start_index: usize,
        end_index: usize,
    },
}

impl Display for TowerRingElement {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match *self {
            TowerRingElement::Face { triangle_index, .. } => {
                write!(f, "F{triangle_index} ")
            }
            TowerRingElement::Edge { end_index, .. } => {
                write!(f, "E{end_index} ")
            }
        }
    }
}

impl Ord for TowerRingElement {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (
                TowerRingElement::Face {
                    triangle_index: s_triangle_index,
                },
                TowerRingElement::Face {
                    triangle_index: o_triangle_index,
                },
            ) => s_triangle_index.cmp(o_triangle_index),
            (
                TowerRingElement::Face { triangle_index },
                TowerRingElement::Edge {
                    start_index,
                    end_index,
                },
            ) => std::cmp::Ordering::Greater,
            (
                TowerRingElement::Edge {
                    start_index,
                    end_index,
                },
                TowerRingElement::Face { triangle_index },
            ) => std::cmp::Ordering::Less,
            (
                TowerRingElement::Edge {
                    start_index: ssi,
                    end_index: sei,
                },
                TowerRingElement::Edge {
                    start_index: osi,
                    end_index: oei,
                },
            ) => ssi.cmp(osi).then(sei.cmp(oei)),
        }
    }
}

impl PartialOrd for TowerRingElement {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for TowerRingElement {
    fn eq(&self, other: &Self) -> bool {
        match self {
            TowerRingElement::Edge {
                end_index,
                start_index,
                ..
            } => match other {
                TowerRingElement::Edge {
                    end_index: oei,
                    start_index: osi,
                    ..
                } => end_index == oei && start_index == osi,
                TowerRingElement::Face { .. } => false,
            },

            TowerRingElement::Face { triangle_index, .. } => match other {
                TowerRingElement::Face {
                    triangle_index: oti,
                    ..
                } => oti == triangle_index,
                TowerRingElement::Edge { .. } => false,
            },
        }
    }
}

impl Hash for TowerRingElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            TowerRingElement::Edge {
                end_index,
                start_index,
                ..
            } => {
                end_index.hash(state);
                start_index.hash(state);
            }
            TowerRingElement::Face { triangle_index, .. } => {
                triangle_index.hash(state);
            }
        }
    }
}

/// Events that occur at vertices during the traversal or slicing of triangles
#[derive(Debug, PartialEq)]
pub enum TriangleEvent {
    /// Represents a vertex that connects two edges within a triangle
    MiddleVertex {
        leading_edge: usize,
        triangle: usize,
        trailing_edge: usize,
    },

    /// Represents an event where a vertex marks the start of a new edge within the triangle
    LeadingEdge {
        leading_edge: usize,
        triangle: usize,
    },

    /// Represents an event where a vertex is the endpoint of an edge within the triangle
    TrailingEdge {
        triangle: usize,
        trailing_edge: usize,
    },
}

/// Takes a list of `TriangleEvent`s and a starting point, for each `TriangleEvent` a face and 1 or 2 edges are created
fn join_triangle_event(events: &[TriangleEvent], starting_point: usize) -> Vec<TowerRing> {
    // log::debug!("Tri events = {:?}", events);
    let mut element_list: Vec<TowerRing> = events
        .iter()
        .map(|event| match event {
            // For `LeadingEdge`s make a face with it's triangle_index
            // and an edge leading_edge as the end_index; then put them in a `TowerRing`
            TriangleEvent::LeadingEdge {
                leading_edge,
                triangle,
            } => {
                let triangle_element = TowerRingElement::Face {
                    triangle_index: *triangle,
                };
                let edge_element = TowerRingElement::Edge {
                    start_index: starting_point,
                    end_index: *leading_edge,
                };

                TowerRing {
                    elements: vec![edge_element, triangle_element],
                }
            }

            // Same as `LeadingEdge` in opasite order
            TriangleEvent::TrailingEdge {
                triangle,
                trailing_edge,
            } => {
                let edge_element = TowerRingElement::Edge {
                    start_index: starting_point,
                    end_index: *trailing_edge,
                };

                let triangle_element = TowerRingElement::Face {
                    triangle_index: *triangle,
                };
                TowerRing {
                    elements: vec![triangle_element, edge_element],
                }
            }

            // Make an edge, face then another edge and put them it a vec
            TriangleEvent::MiddleVertex {
                leading_edge,
                triangle,
                trailing_edge,
            } => {
                let trail_edge_element = TowerRingElement::Edge {
                    start_index: starting_point,
                    end_index: *trailing_edge,
                };

                let triangle_element = TowerRingElement::Face {
                    triangle_index: *triangle,
                };

                let lead_edge_element = TowerRingElement::Edge {
                    start_index: starting_point,
                    end_index: *leading_edge,
                };
                TowerRing {
                    elements: vec![lead_edge_element, triangle_element, trail_edge_element],
                }
            }
        })
        .collect();

    join_fragments(&mut element_list);

    element_list
}

// Join fragmented rings together to for new rings
// A ring can be joined if its last element matches another rings first element
fn join_fragments(fragments: &mut Vec<TowerRing>) {
    //early return for empty fragments
    if fragments.is_empty() {
        return;
    }

    // Sort elements for binary search
    // sorted by the first element in the tower
    // fragments.sort();
    let mut first_pos = fragments.len() - 1;
    while first_pos > 0 {
        //binary search for a matching first element to the current pos last element
        if let Ok(index) = fragments.binary_search_by_key(
            &fragments[first_pos]
                .elements
                .last()
                .expect("Tower rings must contain elements "),
            |a| {
                a.elements
                    .first()
                    .expect("Tower rings must contain elements ")
            },
        ) {
            //Test if this is a complete ring. ie the rings first element and last are indentical
            if index != first_pos {
                // if the removed element is less that the current element the currenly element will be moved by the remove command
                if index < first_pos {
                    first_pos -= 1;
                }

                //remove the ring and join to the current ring
                let removed = fragments.remove(index);
                let first_r = fragments
                    .get_mut(first_pos)
                    .expect("Index is validated by loop ");
                TowerRing::join_rings_in_place(first_r, removed);
            } else {
                // skip already complete elements
                first_pos -= 1;
            }
        } else {
            //if no match is found, move to next element
            first_pos -= 1;
        }
    }
}

pub struct TriangleTowerIterator<'s> {
    /// The [`TriangleTower`] that is iterated through
    pub tower: &'s TriangleTower,
    /// The next vertex in the tower
    pub tower_vert_index: usize,
    /// The **z** height along the slicing *plane*
    plane_height: f64,
    /// The *faces* and *edges* in the tower
    active_rings: Vec<TowerRing>,
    /// The normal vector that defines the slicing plane
    plane_normal: &'s Vertex,
}

/// Calculates the signed projection of a point onto the slicing plane normal.\
/// Returning a scalar value that represents the height of the vertex in the direction of the plane’s normal.
#[inline(always)]
pub fn project_vertex_onto_plane(vertex: &Vertex, plane_normal: &Vertex) -> f64 {
    vertex.dot(plane_normal)
}

impl<'s> TriangleTowerIterator<'s> {
    pub fn new(tower: &'s TriangleTower, plane_normal: &'s Vertex) -> Self {
        // Use the first vertex's projected distance to set the initial plane height
        // todo check if this is correct
        let plane_height = tower.get_height_of_vertex(0, plane_normal);
        Self {
            plane_height,
            tower,
            tower_vert_index: 0,
            active_rings: Vec::new(),
            plane_normal,
        }
    }

    /// Move the iterator up to a specified height along the slicing plane's normal vector
    // todo check fn changes
    pub fn advance_to_height(&mut self, target_height: f64) -> Result<(), SlicerErrors> {
        // round to the secend desimal place
        let target_height = (target_height * 100.0).round() / 100.0;

        let bace_height = self.tower.get_height_of_vertex(0, self.plane_normal);

        println!("current index height: {}\ttarget_height: {}\tagusted target_height: {}", self.tower.get_height_of_vertex(self.tower_vert_index, self.plane_normal), target_height, target_height + bace_height);
        // Iterate up the tower based on the projected height rather than the z-coordinate, target_height is relative to the lowest vertex
        while self.tower.get_height_of_vertex(self.tower_vert_index, self.plane_normal) < target_height + bace_height
            && !self.tower.tower_vertices.is_empty()
        {
            let pop_tower_vert = &self.tower.tower_vertices[self.tower_vert_index];

            // Update active rings by removing edges at the current vertex height
            self.active_rings = self
                .active_rings
                .drain(..)
                .flat_map(|tower_ring| {
                    tower_ring
                        .split_on_edge(pop_tower_vert.start_index)
                        .into_iter()
                })
                .collect();

            self.active_rings.extend(pop_tower_vert.next_ring_fragments.clone());

            join_fragments(&mut self.active_rings);

            // Move to the next vertex in the tower
            self.tower_vert_index += 1;

            // ! why are their so many incomple rings?
            // Check that all ring are complete
            if self.active_rings.iter().any(|ring| { println!("{}", ring); !ring.is_complete_ring()}) {
                return Err(SlicerErrors::TowerGeneration(self.plane_height));
            }
        }

        // Update plane height after advancing
        self.plane_height = target_height;

        Ok(())
    }

    /// Get intersection points of edges with the slicing plane at `plane_height`
    pub fn get_points(&self) -> Vec<Vec<Vertex>> {
        self.active_rings
            .iter()
            .map(|ring| {
                let mut points: Vec<Vertex> = ring
                    .elements
                    .iter()
                    .filter_map(|e| {
                        if let TowerRingElement::Edge { start_index, end_index, .. } = e {
                            Some(plane_intersection(
                                self.plane_height,
                                &self.tower.vertices[*start_index],
                                &self.tower.vertices[*end_index],
                                self.plane_normal,
                            ))
                        } else {
                            None
                        }
                    })
                    .collect();

                // Close the loop by repeating the first point at the end
                if points.first() != points.last() {
                    points.push(points[0].clone());
                }

                points
            })
            .collect()
    }
}

pub fn create_towers(
    models: Vec<(Vec<Vertex>, Vec<IndexedTriangle>)>,
    plane_normal: &Vertex,
) -> Result<Vec<TriangleTower>, SlicerErrors> {
    models
        .into_par_iter()
        .map(|(vertices, triangles)| {
            TriangleTower::from_triangles_and_vertices(&triangles, vertices, plane_normal)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn join_rings(mut first: TowerRing, second: TowerRing) -> TowerRing {
        TowerRing::join_rings_in_place(&mut first, second);

        first
    }

    #[test]
    fn join_rings_test() {
        let r1 = TowerRing {
            elements: vec![
                TowerRingElement::Edge {
                    start_index: 0,
                    end_index: 1,
                },
                TowerRingElement::Face { triangle_index: 0 },
                TowerRingElement::Edge {
                    start_index: 0,
                    end_index: 2,
                },
            ],
        };

        let r2 = TowerRing {
            elements: vec![
                TowerRingElement::Edge {
                    start_index: 0,
                    end_index: 2,
                },
                TowerRingElement::Face { triangle_index: 2 },
                TowerRingElement::Edge {
                    start_index: 4,
                    end_index: 6,
                },
            ],
        };

        let r3 = TowerRing {
            elements: vec![
                TowerRingElement::Edge {
                    start_index: 0,
                    end_index: 1,
                },
                TowerRingElement::Face { triangle_index: 0 },
                TowerRingElement::Edge {
                    start_index: 0,
                    end_index: 2,
                },
                TowerRingElement::Face { triangle_index: 2 },
                TowerRingElement::Edge {
                    start_index: 4,
                    end_index: 6,
                },
            ],
        };

        ring_sliding_equality_assert(&join_rings(r1, r2), &r3);
    }

    #[test]
    fn split_on_edge_test() {
        let r1 = TowerRing {
            elements: vec![
                TowerRingElement::Edge {
                    start_index: 0,
                    end_index: 1,
                },
                TowerRingElement::Face { triangle_index: 0 },
                TowerRingElement::Edge {
                    start_index: 0,
                    end_index: 2,
                },
                TowerRingElement::Face { triangle_index: 2 },
                TowerRingElement::Edge {
                    start_index: 0,
                    end_index: 1,
                },
            ],
        };

        let frags = r1.split_on_edge(2);

        let expected = vec![TowerRing {
            elements: vec![
                TowerRingElement::Face { triangle_index: 2 },
                TowerRingElement::Edge {
                    start_index: 0,
                    end_index: 1,
                },
                TowerRingElement::Face { triangle_index: 0 },
            ],
        }];
        rings_sliding_equality_assert(frags, expected);
    }

    #[test]
    fn assemble_fragment_simple_test() {
        let mut frags = vec![
            TowerRing {
                elements: vec![
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 1,
                    },
                    TowerRingElement::Face { triangle_index: 0 },
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 2,
                    },
                    TowerRingElement::Face { triangle_index: 2 },
                    TowerRingElement::Edge {
                        start_index: 4,
                        end_index: 6,
                    },
                ],
            },
            TowerRing {
                elements: vec![
                    TowerRingElement::Edge {
                        start_index: 4,
                        end_index: 6,
                    },
                    TowerRingElement::Face { triangle_index: 2 },
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 1,
                    },
                ],
            },
        ];

        join_fragments(&mut frags);

        let expected = vec![TowerRing {
            elements: vec![
                TowerRingElement::Edge {
                    start_index: 0,
                    end_index: 1,
                },
                TowerRingElement::Face { triangle_index: 0 },
                TowerRingElement::Edge {
                    start_index: 0,
                    end_index: 2,
                },
                TowerRingElement::Face { triangle_index: 2 },
                TowerRingElement::Edge {
                    start_index: 4,
                    end_index: 6,
                },
                TowerRingElement::Face { triangle_index: 2 },
                TowerRingElement::Edge {
                    start_index: 0,
                    end_index: 1,
                },
            ],
        }];

        rings_sliding_equality_assert(frags, expected);
    }

    #[test]
    fn assemble_fragment_multiple_test() {
        let mut frags = vec![
            TowerRing {
                elements: vec![
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 1,
                    },
                    TowerRingElement::Face { triangle_index: 0 },
                ],
            },
            TowerRing {
                elements: vec![
                    TowerRingElement::Face { triangle_index: 0 },
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 2,
                    },
                    TowerRingElement::Face { triangle_index: 1 },
                ],
            },
            TowerRing {
                elements: vec![
                    TowerRingElement::Face { triangle_index: 1 },
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 3,
                    },
                ],
            },
            TowerRing {
                elements: vec![
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 3,
                    },
                    TowerRingElement::Face { triangle_index: 4 },
                ],
            },
            TowerRing {
                elements: vec![
                    TowerRingElement::Face { triangle_index: 4 },
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 1,
                    },
                ],
            },
            TowerRing {
                elements: vec![
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 11,
                    },
                    TowerRingElement::Face { triangle_index: 10 },
                ],
            },
            TowerRing {
                elements: vec![
                    TowerRingElement::Face { triangle_index: 10 },
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 12,
                    },
                    TowerRingElement::Face { triangle_index: 11 },
                ],
            },
            TowerRing {
                elements: vec![
                    TowerRingElement::Face { triangle_index: 11 },
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 11,
                    },
                ],
            },
        ];

        join_fragments(&mut frags);

        let expected = vec![
            TowerRing {
                elements: vec![
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 1,
                    },
                    TowerRingElement::Face { triangle_index: 0 },
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 2,
                    },
                    TowerRingElement::Face { triangle_index: 1 },
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 3,
                    },
                    TowerRingElement::Face { triangle_index: 4 },
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 1,
                    },
                ],
            },
            TowerRing {
                elements: vec![
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 11,
                    },
                    TowerRingElement::Face { triangle_index: 10 },
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 12,
                    },
                    TowerRingElement::Face { triangle_index: 11 },
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 11,
                    },
                ],
            },
        ];

        rings_sliding_equality_assert(frags, expected);
    }

    #[test]
    fn assemble_fragment_3_fragment_test() {
        let mut frags = vec![
            TowerRing {
                elements: vec![
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 11,
                    },
                    TowerRingElement::Face { triangle_index: 10 },
                ],
            },
            TowerRing {
                elements: vec![
                    TowerRingElement::Face { triangle_index: 10 },
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 12,
                    },
                    TowerRingElement::Face { triangle_index: 11 },
                ],
            },
            TowerRing {
                elements: vec![
                    TowerRingElement::Face { triangle_index: 11 },
                    TowerRingElement::Edge {
                        start_index: 0,
                        end_index: 11,
                    },
                ],
            },
        ];

        join_fragments(&mut frags);

        let expected = vec![TowerRing {
            elements: vec![
                TowerRingElement::Edge {
                    start_index: 0,
                    end_index: 11,
                },
                TowerRingElement::Face { triangle_index: 10 },
                TowerRingElement::Edge {
                    start_index: 0,
                    end_index: 12,
                },
                TowerRingElement::Face { triangle_index: 11 },
                TowerRingElement::Edge {
                    start_index: 0,
                    end_index: 11,
                },
            ],
        }];

        rings_sliding_equality_assert(frags, expected);
    }

    fn rings_sliding_equality_assert(lhs: Vec<TowerRing>, rhs: Vec<TowerRing>) {
        if lhs == rhs {
            return;
        }
        if lhs.len() != rhs.len() {
            panic!("ASSERT rings count are different lengths");
        }

        for q in 0..lhs.len() {
            ring_sliding_equality_assert(&lhs[q], &rhs[q])
        }
    }

    fn ring_sliding_equality_assert(lhs: &TowerRing, rhs: &TowerRing) {
        if lhs == rhs {
            return;
        }
        if lhs.elements.len() != rhs.elements.len() {
            panic!("ASSERT ring {} and {} are different lengths", lhs, rhs);
        }

        for q in 0..lhs.elements.len() - 1 {
            let mut equal = true;
            for w in 0..lhs.elements.len() - 1 {
                equal = equal && rhs.elements[w] == lhs.elements[(w + q) % (lhs.elements.len() - 1)]
            }

            if equal {
                return;
            }

            if lhs.elements.len() != rhs.elements.len() {
                panic!("ASSERT ring {} and {} are different", lhs, rhs);
            }
        }
    }
}
