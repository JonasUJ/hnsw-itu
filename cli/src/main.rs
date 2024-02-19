use std::{path::PathBuf, str::FromStr, time::SystemTime};

use clap::{arg, Parser};
use hdf5::{types::VarLenUnicode, Result};
use hnsw_itu::{Bruteforce, Index, Sketch};
use ndarray::arr1;

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

    #[arg(
        short,
        default_value_t = 10,
        help = "Number of nearest neighbors to find"
    )]
    k: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let dataset = BufferedDataset::open(cli.dataset, "hamming")?;
    let queries = BufferedDataset::open(cli.queries, "hamming")?;

    let mut index = Bruteforce::new();

    let indexing_time = SystemTime::now();
    for sketch in dataset {
        index.add(sketch);
    }
    println!("indexing time: {:?}", indexing_time.elapsed());

    let searching_time = SystemTime::now();
    let queries_vec: Vec<Sketch> = queries.into_iter().collect();
    let results = index.knns(queries_vec.iter().collect(), cli.k);
    println!("search time: {:?}", searching_time.elapsed());

    let knns = BufferedDataset::create("result.h5", (queries_vec.len(), cli.k), "knns")?;

    for (i, res) in results.into_iter().enumerate() {
        let v = arr1(&res.iter().map(|d| d.key() as u64 + 1).collect::<Vec<u64>>());
        knns.write_row(v, i)?;
    }

    knns.add_attr("data", &VarLenUnicode::from_str("hamming").unwrap())?;
    knns.add_attr("size", &VarLenUnicode::from_str("100K").unwrap())?;
    knns.add_attr("algo", &VarLenUnicode::from_str("bruteforce").unwrap())?;
    knns.add_attr("buildtime", &0)?;
    knns.add_attr("querytime", &0)?;
    knns.add_attr("params", &VarLenUnicode::from_str("params").unwrap())?;

    Ok(())
}
