use std::{path::PathBuf, str::FromStr, time::SystemTime};

use clap::{arg, Parser};
use hdf5::{types::VarLenUnicode, File, Result};
use hnsw_itu::{Bruteforce, Index};
use ndarray::{s, Array1};

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
        default_value_t = 100,
        help = "Number of nearest neighbors to find"
    )]
    k: usize,
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
    let mut results = vec![];
    for (i, query) in queries.into_iter().enumerate() {
        if i % 1000 == 0 {
            dbg!(i);
        }

        results.push(index.search(&query, cli.k));
    }
    println!("search time: {:?}", searching_time.elapsed());

    let file = File::create("result.h5")?;
    let knns = file
        .new_dataset::<u64>()
        .shape((10_000, cli.k))
        .create("knns")?;
    let dist = file
        .new_dataset::<u64>()
        .shape((10_000, cli.k))
        .create("dist")?;

    for (i, mut res) in results.into_iter().enumerate() {
        res.sort();

        let v: Vec<u64> = res.iter().map(|d| d.key() as u64 + 1).collect();
        let arr: Array1<u64> = v.into();
        knns.write_slice(arr.view(), s![i, ..])?;

        let v: Vec<u64> = res.iter().map(|d| d.distance() as u64).collect();
        let arr: Array1<u64> = v.into();
        dist.write_slice(arr.view(), s![i, ..])?;
    }

    let _ = file
        .new_attr::<VarLenUnicode>()
        .create("data")?
        .write_scalar(&VarLenUnicode::from_str("hamming").unwrap());
    let _ = file
        .new_attr::<VarLenUnicode>()
        .create("size")?
        .write_scalar(&VarLenUnicode::from_str("100K").unwrap());
    let _ = file
        .new_attr::<VarLenUnicode>()
        .create("algo")?
        .write_scalar(&VarLenUnicode::from_str("bruteforce").unwrap());
    let _ = file
        .new_attr::<usize>()
        .create("buildtime")?
        .write_scalar(&0);
    let _ = file
        .new_attr::<usize>()
        .create("querytime")?
        .write_scalar(&0);
    let _ = file
        .new_attr::<VarLenUnicode>()
        .create("params")?
        .write_scalar(&VarLenUnicode::from_str("params").unwrap());

    Ok(())
}
