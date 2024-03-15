use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
    str::FromStr,
    time::{Duration, SystemTime},
};

use anyhow::{Context, Result};
use bincode::{deserialize_from, serialize_into};
use clap::{arg, Args, Parser, Subcommand, ValueEnum};
use hdf5::{types::VarLenUnicode, File as Hdf5File};
use hnsw_itu::{
    nsw::{NSWBuilder, NSWOptions},
    Bruteforce, Distance, Index, IndexBuilder, Point, NSW,
};
use hnsw_itu_cli::{BufferedDataset, Sketch};
use ndarray::arr1;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, instrument, warn};
use tracing_subscriber::{filter, layer::SubscriberExt, reload, util::SubscriberInitExt};

fn main() -> Result<()> {
    let cli = Cli::parse();
    let level = match cli.verbose.log_level_filter() {
        clap_verbosity_flag::LevelFilter::Off => filter::LevelFilter::OFF,
        clap_verbosity_flag::LevelFilter::Error => filter::LevelFilter::ERROR,
        clap_verbosity_flag::LevelFilter::Warn => filter::LevelFilter::WARN,
        clap_verbosity_flag::LevelFilter::Info => filter::LevelFilter::INFO,
        clap_verbosity_flag::LevelFilter::Debug => filter::LevelFilter::DEBUG,
        clap_verbosity_flag::LevelFilter::Trace => filter::LevelFilter::TRACE,
    };
    let timer = time::format_description::parse("[hour]:[minute]:[second]").unwrap();
    let time_offset = time::UtcOffset::current_local_offset().unwrap_or(time::UtcOffset::UTC);
    let timer = tracing_subscriber::fmt::time::OffsetTime::new(time_offset, timer);
    let (filter, _reload_handle) = reload::Layer::new(level);
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_timer(timer))
        .init();
    debug!(?level, "Logging");
    cli.command.exec()?;
    Ok(())
}

#[instrument(skip_all)]
fn build_index(
    path: &PathBuf,
    algorithm: Algorithm,
    options: impl Into<AlgorithmOptions>,
    start: Option<usize>,
    len: Option<usize>,
) -> Result<IndexFile<Sketch>> {
    info!(?path, "Opening");
    let dataset = BufferedDataset::<'_, Sketch, _>::open(path, "hamming")?;

    let format_size = start.is_none() && len.is_none();
    let skip = start.unwrap_or_default();
    let take = len.unwrap_or(dataset.size());
    let size = take.min(dataset.size() - skip);

    if take != size {
        warn!(
            size,
            len = take,
            "Dataset range will be smaller than specified `len`"
        );
    }

    let mut count = 0;
    let dataset_iter = dataset
        .clone()
        .into_iter()
        .skip(skip)
        .take(take)
        .inspect(|_| {
            count += 1;
            if count % 100000 == 0 {
                debug!(count, "{}%", count * 100 / size);
            }
        });

    let buildtime_start = SystemTime::now();
    let options = options.into();
    info!(
        size,
        ?algorithm,
        single_threaded = options.single_threaded,
        "Building index"
    );
    let index = algorithm.create(dataset_iter, options);
    let buildtime_total = buildtime_start.elapsed().unwrap_or(Duration::ZERO);
    let buildtime_per_element = buildtime_total / size as u32;
    info!(
        "Total build time: {:?}, per element: {:?}",
        buildtime_total, buildtime_per_element
    );

    let attrs = ResultAttrs {
        format_size,
        size,
        algo: algorithm,
        buildtime: buildtime_total.as_secs(),
        ..Default::default()
    };

    Ok(IndexFile { attrs, index })
}

#[instrument(skip_all)]
fn query_index<'a>(
    path: &PathBuf,
    index: &'a Indexes<Sketch>,
    attrs: &mut ResultAttrs,
    k: usize,
    ef: usize,
    single_threaded: bool,
) -> Result<Vec<Vec<Distance<'a, Sketch>>>> {
    if k > ef {
        error!(
            k,
            ef, "`k` is greater than `ef`, this can have adverse effects"
        );
    }

    info!(?path, "Opening");
    let queries = BufferedDataset::open(path, "hamming")?;
    let queries_size: u32 = queries.size().try_into().unwrap();

    let querytime_start = SystemTime::now();
    info!(k, ef, single_threaded, "Start querying");
    let results = if single_threaded {
        queries
            .into_iter()
            .map(|q| index.search(&q, k, ef))
            .collect()
    } else {
        index.knns(queries, k, ef)
    };
    let querytime_total = querytime_start.elapsed().unwrap_or_default();
    let querytime_per_element = querytime_total / queries_size;
    info!(
        "Total query time: {:?}, per query: {:?}",
        querytime_total, querytime_per_element
    );

    attrs.querytime = querytime_total.as_secs();

    Ok(results)
}

#[instrument(skip_all)]
fn read_index(path: &impl AsRef<Path>) -> Result<IndexFile<Sketch>> {
    info!(path = path.as_ref().to_str(), "Reading index");

    let reader = BufReader::new(File::open(path)?);
    let index_file: IndexFile<Sketch> = deserialize_from(reader).context("Could not read index")?;

    info!(size = index_file.index.size(), "Read index");

    Ok(index_file)
}

#[instrument(skip_all)]
fn write_index<P: Serialize>(path: &impl AsRef<Path>, index_file: &IndexFile<P>) -> Result<()> {
    info!(
        path = path.as_ref().to_str(),
        size = index_file.index.size(),
        "Serializing"
    );

    let writer = BufWriter::new(File::create(path)?);
    serialize_into(writer, index_file)?;

    Ok(())
}

fn format_size_string(size: usize) -> String {
    match size {
        90_000..=110_000 => String::from("100K"),
        270_000..=330_000 => String::from("300K"),
        9_000_000..=11_000_000 => String::from("10M"),
        27_000_000..=33_000_000 => String::from("30M"),
        90_000_000..=110_000_000 => String::from("100M"),
        i => i.to_string(),
    }
}

#[instrument(skip_all)]
fn write_result(
    path: &impl AsRef<Path>,
    results: Vec<Vec<Distance<'_, Sketch>>>,
    k: usize,
    sort: bool,
    attrs: ResultAttrs,
) -> Result<()> {
    info!(path = path.as_ref().to_str(), ?sort, "Writing result");
    let knns = BufferedDataset::create(path, (results.len(), k), "knns")?;

    for (i, mut res) in results.into_iter().enumerate() {
        if sort {
            res.sort();
        }

        let v = arr1(&res.iter().map(|d| d.key() as u64 + 1).collect::<Vec<u64>>());
        knns.write_row(v, i)?;
    }

    let size = if attrs.format_size {
        format_size_string(attrs.size)
    } else {
        attrs.size.to_string()
    };

    let data = &VarLenUnicode::from_str(attrs.data.as_str())?;
    let size = &VarLenUnicode::from_str(size.as_str())?;
    let algo = &VarLenUnicode::from_str(format!("{:?}", attrs.algo).as_str())?;
    let params = &VarLenUnicode::from_str(attrs.params.as_str())?;
    info!(
        ?data,
        ?size,
        ?algo,
        buildtime = ?attrs.buildtime,
        querytime = ?attrs.querytime,
        ?params,
        "Writing result attributes"
    );

    knns.add_attr("data", data)?;
    knns.add_attr("size", size)?;
    knns.add_attr("algo", algo)?;
    knns.add_attr("buildtime", &attrs.buildtime)?;
    knns.add_attr("querytime", &attrs.querytime)?;
    knns.add_attr("params", params)?;

    Ok(())
}

#[derive(Serialize, Deserialize, Clone)]
struct ResultAttrs {
    format_size: bool,
    data: String,
    size: usize,
    algo: Algorithm,
    buildtime: u64,
    querytime: u64,
    params: String,
}

impl Default for ResultAttrs {
    fn default() -> Self {
        Self {
            format_size: true,
            data: String::from("hamming"),
            size: Default::default(),
            algo: Default::default(),
            buildtime: Default::default(),
            querytime: Default::default(),
            params: String::from(""),
        }
    }
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(flatten)]
    verbose: clap_verbosity_flag::Verbosity,

    #[command(subcommand)]
    command: Commands,
}

trait Action {
    fn act(self) -> Result<()>;
}

#[derive(Subcommand)]
enum Commands {
    Query(Query),
    Index(CreateIndex),
    QueryIndex(QueryIndex),
    GroundTruth(GroundTruth),
}

impl Commands {
    fn exec(self) -> Result<()> {
        match self {
            Self::Query(a) => a.act(),
            Self::Index(a) => a.act(),
            Self::QueryIndex(a) => a.act(),
            Self::GroundTruth(a) => a.act(),
        }
    }
}

#[derive(Default)]
struct AlgorithmOptions {
    ef_construction: usize,
    connections: usize,
    max_connections: usize,
    single_threaded: bool,
}

#[derive(
    Serialize, Deserialize, Default, Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum,
)]
enum Algorithm {
    #[default]
    Bruteforce,
    Nsw,
}

impl Algorithm {
    fn create<P: Point + Send + Sync>(
        &self,
        dataset: impl IntoIterator<Item = P>,
        options: impl Into<AlgorithmOptions>,
    ) -> Indexes<P> {
        let options = options.into();
        match self {
            Self::Bruteforce => {
                let bruteforce = dataset.into_iter().collect();
                Indexes::Bruteforce(bruteforce)
            }
            Self::Nsw => {
                let iter = dataset.into_iter();
                let mut builder = NSWBuilder::new(NSWOptions {
                    ef_construction: options.ef_construction,
                    connections: options.connections,
                    max_connections: options.max_connections,
                });

                if options.single_threaded {
                    builder.extend(iter)
                } else {
                    builder.extend_parallel(iter);
                }

                Indexes::NSW(builder.build())
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
pub enum Indexes<P> {
    Bruteforce(Bruteforce<P>),
    NSW(NSW<P>),
}

impl<P> Index<P> for Indexes<P> {
    fn size(&self) -> usize {
        match self {
            Self::Bruteforce(bruteforce) => bruteforce.size(),
            Self::NSW(nsw) => nsw.size(),
        }
    }

    fn search<'a>(&'a self, query: &P, k: usize, ef: usize) -> Vec<Distance<'a, P>>
    where
        P: Point,
    {
        match self {
            Self::Bruteforce(bruteforce) => bruteforce.search(query, k, ef),
            Self::NSW(nsw) => nsw.search(query, k, ef),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct IndexFile<P> {
    attrs: ResultAttrs,
    index: Indexes<P>,
}

/// Create index from dataset, query it and generate result file
#[derive(Args, Debug)]
struct Query {
    /// HDF5 file with binary sketches
    #[arg(short, long)]
    datafile: PathBuf,

    /// HDF5 file with queries into the dataset
    #[arg(short = 'Q', long)]
    queryfile: PathBuf,

    /// Location of resulting file
    #[arg(short, long, default_value_t = String::from("result.h5"))]
    outfile: String,

    /// If specified, write index file to this location
    #[arg(short, long)]
    indexfile: Option<PathBuf>,

    /// Number of nearest neighbors to find
    #[arg(short, default_value_t = 10)]
    k: usize,

    /// Beamwidth during search
    #[arg(short = 'e', default_value_t = 48)]
    ef: usize,

    /// Beamwidth during index construction
    #[arg(short = 'c', default_value_t = 48)]
    ef_construction: usize,

    /// Desired number of edges for each node
    #[arg(short = 'm', default_value_t = 48)]
    connections: usize,

    /// Max number of edges for each node
    #[arg(short = 'M', default_value_t = 96)]
    max_connections: usize,

    /// What algorithm to use for index construction
    #[arg(short, long, value_enum, default_value_t = Algorithm::Nsw)]
    algorithm: Algorithm,

    /// Put nearest neighbors in sorted (ascending) order
    #[arg(short, long, default_value_t = false)]
    sort: bool,

    /// Do all querying on a single thread
    #[arg(short = 'S', long, default_value_t = false)]
    single_threaded: bool,
}

impl From<&Query> for AlgorithmOptions {
    fn from(value: &Query) -> Self {
        Self {
            connections: value.connections,
            ef_construction: value.ef_construction,
            max_connections: value.max_connections,
            single_threaded: value.single_threaded,
        }
    }
}

impl Action for Query {
    fn act(self) -> Result<()> {
        let mut index_file = build_index(&self.datafile, self.algorithm, &self, None, None)?;

        if let Some(path) = self.indexfile {
            write_index(&path, &index_file)?;
        }

        let results = query_index(
            &self.queryfile,
            &index_file.index,
            &mut index_file.attrs,
            self.k,
            self.ef,
            self.single_threaded,
        )?;

        write_result(&self.outfile, results, self.k, self.sort, index_file.attrs)?;

        Ok(())
    }
}

/// Index dataset and generate result file used for queries
#[derive(Args, Debug)]
struct CreateIndex {
    /// HDF5 file with binary sketches
    #[arg(short, long)]
    datafile: PathBuf,

    /// Location of resulting file
    #[arg(short, long, default_value_t = String::from("index.idx"))]
    outfile: String,

    /// Beamwidth during index construction
    #[arg(short = 'c', default_value_t = 48)]
    ef_construction: usize,

    /// Desired number of edges for each node
    #[arg(short = 'm', default_value_t = 48)]
    connections: usize,

    /// Max number of edges for each node
    #[arg(short = 'M', default_value_t = 96)]
    max_connections: usize,

    /// At what row in the datafile to start indexing
    #[arg(short = 'b', long)]
    start: Option<usize>,

    /// How many rows from the datafile to index
    #[arg(short, long)]
    len: Option<usize>,

    /// What algorithm to use for index construction
    #[arg(short, long, value_enum, default_value_t = Algorithm::Nsw)]
    algorithm: Algorithm,

    /// Build index on a single thread. Doing so can result in better indexes.
    #[arg(short = 'S', long, default_value_t = false)]
    single_threaded: bool,
}

impl From<&CreateIndex> for AlgorithmOptions {
    fn from(value: &CreateIndex) -> Self {
        Self {
            connections: value.connections,
            ef_construction: value.ef_construction,
            max_connections: value.max_connections,
            single_threaded: value.single_threaded,
        }
    }
}

impl Action for CreateIndex {
    fn act(self) -> Result<()> {
        let index = build_index(&self.datafile, self.algorithm, &self, self.start, self.len)?;
        write_index(&self.outfile, &index)?;

        Ok(())
    }
}

/// Query an index file generated by the `index` command and generate result file
#[derive(Args, Debug)]
struct QueryIndex {
    /// Index file to query
    #[arg(short, long)]
    indexfile: PathBuf,

    /// HDF5 file with queries into the dataset
    #[arg(short = 'Q', long)]
    queryfile: PathBuf,

    /// Location of resulting file
    #[arg(short, long, default_value_t = String::from("result.h5"))]
    outfile: String,

    /// Number of nearest neighbors to find
    #[arg(short, default_value_t = 10)]
    k: usize,

    /// Beamwidth during search
    #[arg(short = 'e', default_value_t = 48)]
    ef: usize,

    /// Put nearest neighbors in sorted (ascending) order
    #[arg(short, long, default_value_t = false)]
    sort: bool,

    /// Do all querying on a single thread
    #[arg(short = 'S', long, default_value_t = false)]
    single_threaded: bool,
}

impl Action for QueryIndex {
    fn act(self) -> Result<()> {
        let mut index_file = read_index(&self.indexfile)?;
        let results = query_index(
            &self.queryfile,
            &index_file.index,
            &mut index_file.attrs,
            self.k,
            self.ef,
            self.single_threaded,
        )?;
        write_result(&self.outfile, results, self.k, self.sort, index_file.attrs)?;

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
    #[arg(short = 'Q', long)]
    queryfile: PathBuf,

    /// Location of resulting file
    #[arg(short, long, default_value_t = String::from("groundtruth.h5"))]
    outfile: String,

    /// At what row in the datafile to start indexing
    #[arg(short = 'b', long)]
    start: Option<usize>,

    /// How many rows from the datafile to index
    #[arg(short, long)]
    len: Option<usize>,

    /// Number of nearest neighbors to find
    #[arg(short, default_value_t = 100)]
    k: usize,

    /// Put nearest neighbors in sorted (ascending) order
    #[arg(short, long, default_value_t = true)]
    sort: bool,
}

impl Action for GroundTruth {
    fn act(self) -> Result<()> {
        let mut index_file = build_index(
            &self.datafile,
            Algorithm::Bruteforce,
            AlgorithmOptions::default(),
            self.start,
            self.len,
        )?;
        let results = query_index(
            &self.queryfile,
            &index_file.index,
            &mut index_file.attrs,
            self.k,
            self.k,
            true,
        )?;

        info!(outfile = self.outfile, sort = self.sort, "Writing result");
        let file = Hdf5File::create(self.outfile)?;
        let knns = BufferedDataset::with_file(&file, (results.len(), self.k), "knns")?;
        let dists = BufferedDataset::with_file(&file, (results.len(), self.k), "dists")?;

        for (i, mut res) in results.into_iter().enumerate() {
            if self.sort {
                res.sort();
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
