#![deny(missing_docs)]

/*!
    Rough algoritim

    build tower
        For each point store all edges and face connected to but above it

    progress up tower
*/

use crate::utils::lerp;
use crate::SlicerErrors;
use binary_heap_plus::{BinaryHeap, MinComparator};
use gladius_shared::types::{IndexedTriangle, Vertex};
use rayon::prelude::*;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

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
fn plane_intersection(
    plane_height: f64,
    v_start: &Vertex,
    v_end: &Vertex,
    normal: &Vertex,
) -> Vertex {
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
pub struct TriangleTower<V: TowerVertex> {
    pub vertices: Vec<V>,
    tower_vertices: BinaryHeap<TowerVertexEvent<V>, MinComparator>,
}

impl<V> TriangleTower<V> where V: TowerVertex {
    /// Create a `TriangleTower` from **vertices** as leading or trailing edges and **triangles**
    /// The normal specifys the slicing plane's normal
    pub fn from_triangles_and_vertices(
        triangles: &[IndexedTriangle],
        vertices: Vec<Vertex>,
        normal: &Vertex,
    ) -> Result<Self, SlicerErrors> {
        let mut future_tower_vert: Vec<Vec<TowerRing>> =
            (0..vertices.len()).map(|_| Vec::new()).collect();

        // for each triangle add it to the tower

        let converted_vertices: Vec<V> = vertices.into_iter().map(|v| V::from_vertex(v)).collect();

        for (triangle_index, index_tri) in triangles.iter().enumerate() {
            // for each edge of the triangle add a fragment to the lower of the points
            for i in 0..3 {
                // if the point edge is rising then the order will be triangle then edge
                // if the edge is falling (or degenerate) it should go edge then triangle

                if converted_vertices[index_tri.verts[i]]
                    < converted_vertices[index_tri.verts[(i + 1) % 3]]
                {
                    let triangle_element = TowerRingElement::Face { triangle_index };
                    let edge_element = TowerRingElement::Edge {
                        start_index: index_tri.verts[i],
                        end_index: index_tri.verts[(i + 1) % 3],
                    };

                    future_tower_vert[index_tri.verts[i]].push(TowerRing {
                        elements: vec![triangle_element, edge_element],
                    });
                } else {
                    let edge_element = TowerRingElement::Edge {
                        start_index: index_tri.verts[(i + 1) % 3],
                        end_index: index_tri.verts[i],
                    };

                    let triangle_element = TowerRingElement::Face { triangle_index };

                    future_tower_vert[index_tri.verts[(i + 1) % 3]].push(TowerRing {
                        elements: vec![edge_element, triangle_element],
                    });
                }
            }
        }

        // for each triangle event, add it to the lowest vertex and
        // create a list of all vertices and there above edges

        let tower_vertices_vec: Vec<TowerVertexEvent<V>> = future_tower_vert
            .into_iter()
            .enumerate()
            .map(|(index, mut fragments)| {
                join_fragments(&mut fragments);
                TowerVertexEvent {
                    start_index: index,
                    next_ring_fragments: fragments,
                    start_vert: converted_vertices
                        .get(index)
                        .expect("validated above")
                        .clone(),
                }
            })
            .collect();

        let mut tower_vertices = BinaryHeap::with_capacity_min(tower_vertices_vec.capacity());

        tower_vertices.extend(tower_vertices_vec);
        Ok(Self {
            vertices: converted_vertices,
            tower_vertices,
        })
    }

    pub fn get_height_of_next_vertex(&self) -> f64 {
        self.tower_vertices
            .peek()
            .map(|vert: &TowerVertexEvent<V>| vert.start_vert.get_height())
            .unwrap_or(f64::INFINITY)
    }
}

/// A vecter of [`TowerRing`]s with a start index, made of triangles
#[derive(Debug)]
struct TowerVertexEvent<V: TowerVertex> {
    pub next_ring_fragments: Vec<TowerRing>,
    pub start_index: usize,
    pub start_vert: V,
}

impl<V: TowerVertex> PartialOrd for TowerVertexEvent<V> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<V: TowerVertex> Ord for TowerVertexEvent<V> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.start_vert
            .partial_cmp(&other.start_vert)
            .expect("NO_NAN")
    }
}

impl<V: TowerVertex> Eq for TowerVertexEvent<V> {}

impl<V> PartialEq for TowerVertexEvent<V>
where
    V: TowerVertex + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.start_vert.eq(&other.start_vert)
    }
}

/// A list of **faces** and **edges**\
/// When complete it will have at least 3 [`TowerRingElement`]s and equal first and last elements.\
/// This needs to be checked with [`TowerRing::is_complete_ring`]
#[derive(Clone, Debug, PartialEq, Eq)]
struct TowerRing {
    /// The elements that make up the ring
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
    /// A triangle
    Face {
        /// The index of the tri
        triangle_index: usize,
    },
    /// A line connecting two vertices
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

// Join fragmented rings together to for new rings
// A ring can be joined if its last element matches another rings first element
fn join_fragments(fragments: &mut Vec<TowerRing>) {
    //early return for empty fragments
    if fragments.is_empty() {
        return;
    }

    // Sort elements for binary search
    // sorted by the first element in the tower
    // todo check if sort is needed
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
            // Test if this is a complete ring. ie the rings first element and last are identical
            if index != first_pos {
                // if the removed element is less that the current element the currently element will be moved by the remove command
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

/// A struct that helps iterate through a [`TriangleTower`] by holding information about were it is in the tower
pub struct TriangleTowerIterator<'s, F: TowerVertex> {
    /// The [`TriangleTower`] that is iterated through
    tower: TriangleTower<F>,
    /// The next vertex in the tower
    tower_vert_index: usize,
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
pub fn project_vertex_onto_plane<V: TowerVertex>(vertex: &V, plane_normal: &Vertex) -> f64 {
    vertex.dot(plane_normal)
}

impl<'s, V> TriangleTowerIterator<'s, V> where V: TowerVertex {
    pub fn new(tower: TriangleTower<V>, plane_normal: &'s Vertex) -> Self {
        let plane_height = tower.get_height_of_next_vertex();
        Self {
            plane_height,
            tower,
            active_rings: Vec::new(),
            tower_vert_index: 0,
            plane_normal,
        }
    }

    /// Move the iterator up to a specified height along the slicing plane's normal vector
    pub fn advance_to_height(&mut self, z: f64) -> Result<(), SlicerErrors> {
        while self.tower.get_height_of_next_vertex() < z && !self.tower.tower_vertices.is_empty() {
            let pop_tower_vert = self.tower.tower_vertices.pop().expect("Validated above");

            // Create Frags from rings by removing current edges
            self.active_rings = self
                .active_rings
                .drain(..)
                .flat_map(|tower_ring| {
                    tower_ring
                        .split_on_edge(pop_tower_vert.start_index)
                        .into_iter()
                })
                .collect();

            self.active_rings.extend(pop_tower_vert.next_ring_fragments);

            join_fragments(&mut self.active_rings);

            self.tower_vert_index += 1;

            for ring in &self.active_rings {
                if !ring.is_complete_ring() {
                    return Err(SlicerErrors::TowerGeneration(self.plane_height));
                }
            }
        }

        self.plane_height = z;

        Ok(())
    }

    pub fn get_points(&self) -> Vec<Vec<V>> {
        self.active_rings
            .iter()
            .map(|ring| {
                let mut points: Vec<V> = ring
                    .elements
                    .iter()
                    .filter_map(|e| {
                        if let TowerRingElement::Edge {
                            start_index,
                            end_index,
                            ..
                        } = e {
                            Some(V::line_height_intersection(
                                self.plane_height,
                                &self.tower.vertices[*start_index],
                                &self.tower.vertices[*end_index],
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

/// Creates a `TriangleTower` for each model in parallel
pub fn create_towers<V: TowerVertex>(
    models: Vec<(Vec<Vertex>, Vec<IndexedTriangle>)>,
    plane_normal: &Vertex,
) -> Result<Vec<TriangleTower<V>>, SlicerErrors> {
    models
        .into_par_iter()
        .map(|(vertices, triangles)| {
            TriangleTower::from_triangles_and_vertices(&triangles, vertices, plane_normal)
        })
        .collect()
}

/// Used to allow vertices with custom ord impls'
pub trait TowerVertex: Clone + Eq + Ord + Send {
    /// Convert from vertex to this type
    fn from_vertex(vertex: Vertex) -> Self;

    /// Return the height of this vertex
    fn get_height(&self) -> f64;

    /// Gets the x position for slicing purposes
    fn get_slice_x(&self) -> f64;

    /// Gets the y position for slicing purposes
    fn get_slice_y(&self) -> f64;

    /// Gets the vertex at the specified height between start and end
    fn line_height_intersection(height: f64, v_start: &Self, v_end: &Self) -> Self;

    /// Get the dot product of two tower vertices
    #[inline]
    fn dot<V: TowerVertex>(&self, other: &V) -> f64 {
        self.get_slice_x() * other.get_slice_x()
        +
        self.get_slice_y() * other.get_slice_y()
        +
        self.get_height() * other.get_height()
    }
}

impl TowerVertex for Vertex {
    #[inline]
    fn from_vertex(vertex: Vertex) -> Self {
        // No type conversion needed
        vertex
    }

    #[inline]
    fn get_height(&self) -> f64 {
        // height is just Z position
        self.z
    }

    fn line_height_intersection(height: f64, v_start: &Vertex, v_end: &Vertex) -> Vertex {
        // Lerps from start to end given the height
        let z_normal = (height - v_start.z) / (v_end.z - v_start.z);
        debug_assert!(z_normal <= 1.0);

        let y = lerp(v_start.y, v_end.y, z_normal);
        let x = lerp(v_start.x, v_end.x, z_normal);
        Vertex { x, y, z: height }
    }

    #[inline]
    fn get_slice_x(&self) -> f64 {
        // slice just uses x position
        self.x
    }

    #[inline]
    fn get_slice_y(&self) -> f64 {
        // slice just uses y position
        self.y
    }
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
