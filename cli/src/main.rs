use std::{path::PathBuf, time::SystemTime};

use clap::{arg, Parser};
use hdf5::Result;
use hnsw_itu::{Bruteforce, Index};

#[allow(dead_code)]
mod dataset;
use crate::dataset::*;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(value_name = "DATASET", help = "HDF5 file with binary sketches")]
    dataset: PathBuf,

    #[arg(value_name = "QUERIES", help = "HDF5 file with queries into <DATASET>")]
    queries: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let dataset = SketchDataset::create(cli.dataset, "hamming")?;
    let queries = SketchDataset::create(cli.queries, "hamming")?;

    let mut index = Bruteforce::new();

    let indexing_time = SystemTime::now();
    for sketch in dataset {
        index.add(sketch);
    }
    println!("indexing time: {:?}", indexing_time.elapsed());

    let searching_time = SystemTime::now();
    for (i, query) in queries.into_iter().enumerate() {
        if i % 100 == 0 {
            dbg!(i);
        }

        index.search(&query, 100);
    }
    println!("search time: {:?}", searching_time.elapsed());

    Ok(())
}
