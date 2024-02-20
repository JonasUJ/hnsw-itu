use std::{
    path::PathBuf,
    str::FromStr,
    time::{Duration, SystemTime},
};

use clap::{arg, Args, Parser, Subcommand};
use hdf5::{types::VarLenUnicode, File, Result};
use hnsw_itu::{Bruteforce, Index};
use ndarray::arr1;

mod dataset;
mod sketch;

use crate::dataset::*;
use crate::sketch::*;

fn main() -> Result<()> {
    let cli = Cli::parse();
    cli.command.exec()?;
    Ok(())
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

trait Action {
    fn act(self) -> Result<()>;
}

#[derive(Subcommand)]
enum Commands {
    Query(Query),
    GroundTruth(GroundTruth),
}

impl Commands {
    fn exec(self) -> Result<()> {
        match self {
            Self::GroundTruth(a) => a.act(),
            Self::Query(a) => a.act(),
        }
    }
}

/// Query dataset and generate result file
#[derive(Args)]
struct Query {
    /// HDF5 file with binary sketches
    #[arg(short, long)]
    datafile: PathBuf,

    /// HDF5 file with queries into the dataset
    #[arg(short, long)]
    queryfile: PathBuf,

    /// Location of resulting file
    #[arg(short, long, default_value_t = String::from("result.h5"))]
    outfile: String,

    /// Number of nearest neighbors to find
    #[arg(short, default_value_t = 10)]
    k: usize,

    /// Put nearest neighbors in sorted (ascending) order
    #[arg(short, long, default_value_t = false)]
    sort: bool,
}

impl Action for Query {
    fn act(self) -> Result<()> {
        let dataset = BufferedDataset::open(self.datafile, "hamming")?;
        let queries = BufferedDataset::<'_, Sketch, _>::open(self.queryfile, "hamming")?;

        let buildtime = SystemTime::now();
        let index = Bruteforce::from_iter(dataset);
        let buildtime_sec = buildtime.elapsed().unwrap_or(Duration::ZERO).as_secs();

        let querytime = SystemTime::now();
        let results = index.knns(queries.clone(), self.k);
        let querytime_sec = querytime.elapsed().unwrap_or(Duration::ZERO).as_secs();

        let knns = BufferedDataset::create(self.outfile, (queries.size(), self.k), "knns")?;

        for (i, mut res) in results.into_iter().enumerate() {
            if self.sort {
                res.sort()
            }

            let v = arr1(&res.iter().map(|d| d.key() as u64 + 1).collect::<Vec<u64>>());
            knns.write_row(v, i)?;
        }

        let size = match index.size() {
            90_000..=110_000 => String::from("100K"),
            270_000..=330_000 => String::from("300K"),
            9_000_000..=11_000_000 => String::from("10M"),
            27_000_000..=33_000_000 => String::from("30M"),
            90_000_000..=110_000_000 => String::from("100M"),
            i => i.to_string(),
        };

        knns.add_attr("data", &VarLenUnicode::from_str("hamming").unwrap())?;
        knns.add_attr("size", &VarLenUnicode::from_str(size.as_str()).unwrap())?;
        knns.add_attr("algo", &VarLenUnicode::from_str("bruteforce").unwrap())?;
        knns.add_attr("buildtime", &buildtime_sec)?;
        knns.add_attr("querytime", &querytime_sec)?;
        knns.add_attr("params", &VarLenUnicode::from_str("params").unwrap())?;

        Ok(())
    }
}

/// Generate ground truth from a dataset given a set of queries
#[derive(Args)]
struct GroundTruth {
    /// HDF5 file with binary sketches
    #[arg(short, long)]
    datafile: PathBuf,

    /// HDF5 file with queries into the dataset
    #[arg(short, long)]
    queryfile: PathBuf,

    /// Location of resulting file
    #[arg(short, long, default_value_t = String::from("groundtruth.h5"))]
    outfile: String,

    /// Number of nearest neighbors to find
    #[arg(short, default_value_t = 100)]
    k: usize,

    /// Put nearest neighbors in sorted (ascending) order
    #[arg(short, long, default_value_t = true)]
    sort: bool,
}

impl Action for GroundTruth {
    fn act(self) -> Result<()> {
        let dataset = BufferedDataset::open(self.datafile, "hamming")?;
        let queries = BufferedDataset::<'_, Sketch, _>::open(self.queryfile, "hamming")?;

        let index = Bruteforce::from_iter(dataset);

        let results = index.knns(queries.clone(), self.k);

        let file = File::create(self.outfile)?;
        let knns = BufferedDataset::with_file(&file, (queries.size(), self.k), "knns")?;
        let dists = BufferedDataset::with_file(&file, (queries.size(), self.k), "dists")?;

        for (i, mut res) in results.into_iter().enumerate() {
            if self.sort {
                res.sort()
            }

            let (nn, dist): (Vec<_>, Vec<_>) = res
                .iter()
                .map(|d| (d.key() as u64 + 1, d.distance() as u64))
                .unzip();

            knns.write_row(arr1(&nn), i)?;
            dists.write_row(arr1(&dist), i)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_cli() {
        use clap::CommandFactory;
        Cli::command().debug_assert()
    }
}
