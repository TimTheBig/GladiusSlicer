#![deny(clippy::unwrap_used, clippy::perf)]
#![warn(clippy::all, clippy::missing_const_for_fn)]

use clap::Parser;
use gladius_shared::loader::{Loader, STLLoader, ThreeMFLoader};
use gladius_shared::types::*;
use input::load_settings;
use utils::{DisplayType, StateContext};

use crate::plotter::convert_objects_into_moves;
use crate::tower::{create_towers, TriangleTower, TriangleTowerIterator};
use geo::{
    coordinate_position, Closest, ClosestPoint, Contains, Coord, CoordinatePosition, GeoFloat,
    Line, MultiPolygon, Point,
};
use gladius_shared::settings::{PartialSettingsFile, Settings, SettingsValidationResult};
use std::fs::File;

use std::ffi::OsStr;
use std::path::Path;

use crate::bounds_checking::{check_model_bounds, check_moves_bounds};
use crate::calculation::calculate_values;
use crate::command_pass::{CommandPass, OptimizePass, SlowDownLayerPass};
use crate::converter::convert;
use crate::plotter::polygon_operations::PolygonOperations;
use crate::slice_pass::*;
use crate::slicing::slice;
use crate::utils::{
    send_error_message, send_warning_message, show_error_message, show_warning_message,
    state_update,
};
use gladius_shared::error::SlicerErrors;
use gladius_shared::messages::Message;
use itertools::Itertools;
use log::{debug, info, LevelFilter};
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use simple_logger::SimpleLogger;
use std::collections::HashMap;
use std::io::BufWriter;

mod bounds_checking;
mod calculation;
mod command_pass;
mod converter;
mod input;
mod optimizer;
mod plotter;
mod slice_pass;
mod slicing;
mod test;
mod tower;
mod utils;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[clap(group(
    clap::ArgGroup::new("settings-group")
        .required(true)
        .args(&["settings_file_path", "settings_json"]),
))]
struct Args {
    #[arg(
        required = true,
        help = "The input files and there translations.\nBy default it takes a list of json strings that represents how the models should be loaded and translated.\nSee simple_input for an alternative command. "
    )]
    input: Vec<String>,

    #[arg(short = 'o', help = "Sets the output dir")]
    output: Option<String>,

    #[arg(short = 'v', action = clap::ArgAction::Count, conflicts_with = "message", help = "Sets the level of verbosity")]
    verbose: u8,

    #[arg(short = 's', help = "Sets the settings file to use")]
    settings_file_path: Option<String>,
    #[arg(short = 'S', help = "The contents of a json settings file.")]
    settings_json: Option<String>,
    #[arg(short = 'm', help = "Use the Message System (useful for interprocess communication)")]
    message: bool,

    #[arg(
        long = "print_settings",
        conflicts_with = "message",
        help = "Print the final combined settings out to Stdout and Terminate. Verbose level 4 will print but continue."
    )]
    print_settings: bool,
    #[arg(
        long = "simple_input",
        alias = "simple",
        help = "The input should only be a list of files that will be auto translated to the center of the build plate."
    )]
    simple_input: bool,
    #[arg(
        short = 'j',
        help = "Sets the number of threads to use in the thread pool (defaults to number of CPUs)"
    )]
    thread_count: Option<usize>,
}

fn main() {
    #[cfg(feature = "json_schema_gen")]
    // export json schema for settings
    Settings::gen_schema(Path::new("settings/")).expect("The programme should exit if this fails");

    // The YAML file is found relative to the current file, similar to how modules are found
    let args: Args = Args::parse();

    // set number of cores for rayon
    if let Some(number_of_threads) = args.thread_count {
        rayon::ThreadPoolBuilder::new()
            .num_threads(number_of_threads)
            .build_global()
            .expect("Only call to build global");
    }

    let mut state_context = StateContext::new(if args.message {
        DisplayType::Message
    } else {
        DisplayType::StdOut
    });

    if !args.message {
        // Vary the output based on how many times the user used the "verbose" flag
        // (i.e. 'myprog -v -v -v' or 'myprog -vvv' vs 'myprog -v'

        SimpleLogger::new()
            .with_level(match args.verbose {
                0 => LevelFilter::Error,
                1 => LevelFilter::Warn,
                2 => LevelFilter::Info,
                3 => LevelFilter::Debug,
                _ => LevelFilter::Trace,
            })
            .init()
            .expect("Only Logger Setup");
    }

    state_update("Loading Inputs", &mut state_context);

    let settings_json = args.settings_json.unwrap_or_else(|| {
        handle_err_or_return(
            input::load_settings_json(
                args.settings_file_path
                    .as_deref()
                    .expect("CLAP should handle requiring a settings option to be Some"),
            ),
            &state_context,
        )
    });

    let settings = handle_err_or_return(
        load_settings(args.settings_file_path.as_deref(), &settings_json),
        &state_context,
    );

    let models = handle_err_or_return(
        crate::input::load_models(Some(args.input), &settings, args.simple_input),
        &state_context,
    );
    if args.print_settings {
        for line in gladius_shared::settings::SettingsPrint::to_strings(&settings) {
            println!("{}", line);
        }
        std::process::exit(0);
    } else if log::log_enabled!(log::Level::Trace) {
        for line in gladius_shared::settings::SettingsPrint::to_strings(&settings) {
            log::trace!("{}", line);
        }
    }

    handle_err_or_return(check_model_bounds(&models, &settings), &state_context);

    handle_setting_validation(settings.validate_settings(), &state_context);

    // Calculate the normal
    let plane_normal = angle_to_normal(settings.slice_angle.unwrap_or_default());
    // !
    println!(
        "{:?}",
        verts_per_layer_test(models[0].0.clone(), &plane_normal)
    );
    // !
    let towers: Vec<TriangleTower<_>> = handle_err_or_return(
        create_towers::<Vertex>(models, &plane_normal),
        &state_context,
    );
    state_update("Creating Towers", &mut state_context);

    let objects = handle_err_or_return(slice(towers, &settings, &plane_normal), &state_context);
    state_update("Slicing", &mut state_context);

    state_update("Generating Moves", &mut state_context);

    let mut moves = handle_err_or_return(
        generate_moves(objects, &settings, &mut state_context),
        &state_context,
    );

    handle_err_or_return(check_moves_bounds(&moves, &settings), &state_context);

    state_update("Optimizing", &mut state_context);
    debug!("Optimizing {} Moves", moves.len());

    OptimizePass::pass(&mut moves, &settings);
    state_update("Slowing Layer Down", &mut state_context);

    SlowDownLayerPass::pass(&mut moves, &settings);

    if let DisplayType::Message = state_context.display_type {
        let message = Message::Commands(moves.clone());
        bincode::serialize_into(BufWriter::new(std::io::stdout()), &message)
            .expect("Write Limit should not be hit");
    }
    state_update("Calculate Values", &mut state_context);

    // Display info about the print
    print_info_message(&state_context, &moves, &settings);

    state_update("Outputting G-code", &mut state_context);

    // Output the GCode
    if let Some(file_path) = &args.output {
        // Output to file
        debug!("Converting {} Moves", moves.len());
        handle_err_or_return(
            convert(
                &moves,
                &settings,
                &mut handle_err_or_return(
                    File::create(file_path).map_err(|_| SlicerErrors::FileCreateError {
                        filepath: file_path.to_string(),
                    }),
                    &state_context,
                ),
            ),
            &state_context,
        );
    } else {
        match state_context.display_type {
            DisplayType::Message => {
                // Output as message
                let mut gcode: Vec<u8> = Vec::new();
                handle_err_or_return(convert(&moves, &settings, &mut gcode), &state_context);
                let message = Message::GCode(
                    String::from_utf8(gcode)
                        .expect("All write occur from write macro so should be utf8"),
                );
                bincode::serialize_into(BufWriter::new(std::io::stdout()), &message)
                    .expect("Write Limit should not be hit");
            }
            DisplayType::StdOut => {
                // Output to stdout
                let stdout = std::io::stdout();
                let mut stdio_lock = stdout.lock();
                debug!("Converting {} Moves", moves.len());
                handle_err_or_return(convert(&moves, &settings, &mut stdio_lock), &state_context);
            }
        }
    };

    if let DisplayType::StdOut = state_context.display_type {
        info!(
            "Total slice time {} msec",
            state_context.get_total_elapsed_time().as_millis()
        );
    }
}

fn angle_to_normal(slice_angle: f64) -> Vertex {
    println!("{}", slice_angle);
    // Convert slice angle from degrees to radians
    let slice_angle_radians = slice_angle * std::f64::consts::PI / 180.0;

    // Calculate the normal vector based on the angle
    // todo in XZ plane mode switch x and y then z and y
    let plane_normal = Vertex {
        x: 1.0, // slice_angle_radians.cos(),
        y: 0.0,
        z: 1.0, // slice_angle_radians.sin(),
    };
    println!("{:?}", plane_normal);
    plane_normal
}

fn verts_per_layer_test<V: tower::TowerVertex>(verts: Vec<V>, plane_normal: &Vertex) -> Vec<u32> {
    // Sort tower vertices based on their projection along the normal vector
    // verts.sort_by(|a, b| {
    //     // Calculate projection for vertces
    //     a.dot(plane_normal)
    //         .partial_cmp(&b.dot(plane_normal))
    //         .expect("STL ERROR: No Points should have NAN values")
    // });

    // todo test use of plane_normal
    let mut verts_per_layer: Vec<u32> = Vec::new();
    for vertex in &verts {
        println!(
            "project_vertex: {}",
            tower::project_vertex_onto_plane(vertex, plane_normal)
        );
        println!(
            "project_vertex usize: {}",
            tower::project_vertex_onto_plane(vertex, plane_normal)
                .abs()
                .round() as usize
        );
        match verts_per_layer.get_mut(
            tower::project_vertex_onto_plane(vertex, plane_normal)
                .abs()
                .round() as usize,
        ) {
            Some(c) => *c += 1,
            // if none the val with be next
            None => {
                verts_per_layer.push(1);
            }
        };
    }
    // 45
    verts_per_layer
}

/// Display info about the print; time and filament info
fn print_info_message(state_context: &StateContext, moves: &[Command], settings: &Settings) {
    let cv = calculate_values(moves, settings);

    match state_context.display_type {
        DisplayType::Message => {
            let message = Message::CalculatedValues(cv);
            bincode::serialize_into(BufWriter::new(std::io::stdout()), &message)
                .expect("Write Limit should not be hit");
        }
        DisplayType::StdOut => {
            let (hour, min, sec, _) = cv.get_hours_minutes_seconds_fract_time();

            info!(
                "Total Time: {} hours {} minutes {:.3} seconds",
                hour, min, sec
            );
            info!(
                "Total Filament Volume: {:.3} cm^3",
                cv.plastic_volume / 1000.0
            );
            info!("Total Filament Mass: {:.3} grams", cv.plastic_weight);
            info!(
                "Total Filament Length: {:.3} meters",
                cv.plastic_length / 1000.0
            );
            info!(
                "Total Filament Cost: ${:.2}",
                (((cv.plastic_volume / 1000.0) * settings.filament.density) / 1000.0)
                    * settings.filament.cost
            );
        }
    }
}

fn generate_moves(
    mut objects: Vec<Object>,
    settings: &Settings,
    state_context: &mut StateContext,
) -> Result<Vec<Command>, SlicerErrors> {
    // Creates Support Towers
    SupportTowerPass::pass(&mut objects, settings, state_context);

    // Adds a skirt
    SkirtPass::pass(&mut objects, settings, state_context);

    // Adds a brim
    BrimPass::pass(&mut objects, settings, state_context);

    let v: Result<Vec<()>, SlicerErrors> = objects
        .iter_mut()
        .map(|object| {
            let slices = &mut object.layers;

            // Shrink layer
            ShrinkPass::pass(slices, settings, state_context)?;

            // Handle Perimeters
            PerimeterPass::pass(slices, settings, state_context)?;

            // Handle Bridging
            BridgingPass::pass(slices, settings, state_context)?;

            // Handle Top Layer
            TopLayerPass::pass(slices, settings, state_context)?;

            // Handle Top And Bottom Layers
            TopAndBottomLayersPass::pass(slices, settings, state_context)?;

            // Handle Support
            SupportPass::pass(slices, settings, state_context)?;

            // Lightning Infill
            LightningFillPass::pass(slices, settings, state_context)?;

            // Fill Remaining areas
            FillAreaPass::pass(slices, settings, state_context)?;

            // Order the move chains
            OrderPass::pass(slices, settings, state_context)
        })
        .collect();

    v?;

    Ok(convert_objects_into_moves(objects, settings))
}

fn handle_err_or_return<T>(res: Result<T, SlicerErrors>, state_context: &StateContext) -> T {
    match res {
        Ok(data) => data,
        Err(slicer_error) => {
            match state_context.display_type {
                DisplayType::Message => send_error_message(slicer_error),
                DisplayType::StdOut => show_error_message(&slicer_error),
            }
            std::process::exit(-1);
        }
    }
}

/// Sends an apropreate error/warning message for a `SettingsValidationResult`
fn handle_setting_validation(res: SettingsValidationResult, state_context: &StateContext) {
    match res {
        SettingsValidationResult::NoIssue => {}
        SettingsValidationResult::Warning(slicer_warning) => match state_context.display_type {
            DisplayType::Message => send_warning_message(slicer_warning),
            DisplayType::StdOut => show_warning_message(&slicer_warning),
        },
        SettingsValidationResult::Error(slicer_error) => {
            match state_context.display_type {
                DisplayType::Message => send_error_message(slicer_error),
                DisplayType::StdOut => show_error_message(&slicer_error),
            }
            std::process::exit(-1);
        }
    }
}

#[cfg(test)]
mod test_main {
    use super::*;

    #[test]
    fn test_angle_to_normal() {
        assert_ne!(angle_to_normal(0.0), angle_to_normal(45.0));
        assert_eq!(angle_to_normal(2.0), angle_to_normal(2.0))
        // todo improve
    }

    #[test]
    fn verts_per_layer() {
        let verts = vec![
            Vertex {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            Vertex {
                x: 1.0,
                y: 0.0,
                z: 1.0,
            },
            Vertex {
                x: 2.0,
                y: 0.0,
                z: 2.0,
            },
            Vertex {
                x: 3.0,
                y: 0.0,
                z: 3.0,
            },
            Vertex {
                x: 4.0,
                y: 0.0,
                z: 4.0,
            },
            Vertex {
                x: 5.0,
                y: 0.0,
                z: 5.0,
            },
            Vertex {
                x: 6.0,
                y: 0.0,
                z: 6.0,
            },
            Vertex {
                x: 7.0,
                y: 0.0,
                z: 7.0,
            },
            Vertex {
                x: 8.0,
                y: 0.0,
                z: 8.0,
            },
            Vertex {
                x: 9.0,
                y: 0.0,
                z: 9.0,
            },
            Vertex {
                x: 90.5,
                y: 0.0,
                z: 9.0,
            },
        ];

        assert_eq!(
            verts_per_layer_test(
                verts,
                &Vertex {
                    x: -1.0,
                    y: 0.0,
                    z: 1.0
                }
            ),
            vec![10, 1]
        );
    }
}
