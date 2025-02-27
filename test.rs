use geo_3d::Coord;

pub trait TowerVertex: Ord + Send + Eq {
    /// Gets the z, **not** height, of the vertex
    fn get_z(&self) -> f64;

    /// Return the height of this vertex
    fn get_height(&self) -> f64;

    /// Gets the x position for slicing purposes
    fn get_slice_x(&self) -> f64;

    /// Gets the y position for slicing purposes
    fn get_slice_y(&self) -> f64;

    /// Get the dot product of two tower vertices
    /// This must not use get_height as that would be an **∞** loop
    #[inline]
    fn dot<V: TowerVertex>(&self, other: &V) -> f64 {
        self.get_slice_x() * other.get_slice_x()
            + self.get_slice_y() * other.get_slice_y()
            + self.get_z() * other.get_z()
    }
}

/// A single 3D vertex, with normal vertex perjection based methods
#[derive(Default, Clone, Debug, PartialEq)]
pub struct NormalVertex {
    /// X Coord
    pub x: f64,

    /// Y Coord
    pub y: f64,

    /// Z Coord
    pub z: f64,
}

impl Eq for NormalVertex {}

impl Ord for NormalVertex {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // get the normal to project on for z
        let normal = PLANE_NORMAL.get()
            .expect("This is initialized before this can be called in main");

        let projected_cmp = self.dot(normal)
            .partial_cmp(&other.dot(normal))
            .expect("Non-NAN");

        if projected_cmp != std::cmp::Ordering::Equal {
            projected_cmp
        } else if self.z != other.z {
            self.z.partial_cmp(&other.z).expect("Non-NAN")
        } else if self.y != other.y {
            self.y.partial_cmp(&other.y).expect("Non-NAN")
        } else {
            self.x.partial_cmp(&other.x).expect("Non-NAN")
        }
    }
}

impl PartialOrd for NormalVertex {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl TowerVertex for NormalVertex {
    fn get_z(&self) -> f64 {
        self.z
    }

    fn get_height(&self) -> f64 {
        // height is the Z position perjected on to the plane normal
        self.dot(
            PLANE_NORMAL.get()
                .expect("This is initialized before this can be called in main"),
        )
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

impl NormalVertex {
    // todo check
    /// Project the 3D vertex onto a 2D plane, defined by its normal vector `PLANE_NORMAL` and an offset.
    /// This allows the plotter to run normally.
    pub(crate) fn project_on_plane_to_2d(self) -> Coord<f64> {
        let plane_normal = PLANE_NORMAL.get()
            .expect("This is initialized before this can be called in main");

        // Define the offset of the plane from the origin
        let plane_offset = 0.0;

        // Calculate the scalar distance from the point to the plane
        let distance = (plane_normal.dot(&self) + plane_offset)
            / (plane_normal.x.powi(2) + plane_normal.y.powi(2) + plane_normal.z.powi(2)).sqrt();

        // Subtract the distance along the plane normal to get the projection
        let projected_x = self.x - distance * plane_normal.x;
        let projected_y = self.y - distance * plane_normal.y;
        let projected_z = self.z - distance * plane_normal.z;

        // For 2D, we use the `x` and `z` coordinates (you can adjust this based on your needs)
        Coord {
            x: projected_x,
            y: projected_y, // Use Z as the 2nd coordinate in the 2D system
        }
    }
}

/// Convert the slice angle from degrees to radians, then calculate the normal vector based on the angle
pub fn angle_to_normal(slice_angle: f64) -> NormalVertex {
    // Convert slice angle from degrees to radians
    let slice_angle_radians = slice_angle * std::f64::consts::PI / 180.0;

    // Calculate the normal vector based on the angle
    // In XZ plane mode switch x and z then z and y
    // plane_normal
    NormalVertex {
        // 0.0 when angle is zero, 0.7071067812 when angle is 45.0
        x: slice_angle_radians.sin(),
        // not involved in angled slicing
        y: 0.0,
        // 1.0 when angle is zero, so dot product is z, 0.7071067812 when angle is 45.0
        z: slice_angle_radians.cos(),
    }
}

pub static PLANE_NORMAL: std::sync::OnceLock<NormalVertex> = std::sync::OnceLock::new();

fn main() {
    PLANE_NORMAL.get_or_init(|| angle_to_normal(45.0) );
    println!("{:?}", NormalVertex { x: 10.0, y: 10.0, z: 1.0 }.project_on_plane_to_2d())
}
